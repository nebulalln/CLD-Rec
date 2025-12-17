import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cld_stage1 import CLDStage1


class Projector(nn.Module):
    """
    Projector module: Maps Stage-1 embeddings to LLM hidden space.
    Architecture: Linear -> GeLU -> Linear
    """
    def __init__(self, embedding_dim: int = 50, llm_hidden_size: int = 2048):
        """
        Args:
            embedding_dim: Dimension of Stage-1 embeddings (default: 50)
            llm_hidden_size: Hidden size of LLM (Qwen1.5-1.8B: 2048)
        """
        super(Projector, self).__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projector weights."""
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings to LLM hidden space.
        
        Args:
            embeddings: Input embeddings of shape (..., embedding_dim)
        
        Returns:
            Projected embeddings of shape (..., llm_hidden_size)
        """
        return self.projector(embeddings)


class CLDStage2(nn.Module):
    """
    CLD-Rec Stage 2: LLM Integration Model
    
    This stage integrates debiased collaborative knowledge from Stage 1 with LLMs.
    Only the Projector is trainable; Stage-1 and LLM are frozen.
    """
    
    def __init__(self, stage1_model: CLDStage1, llm_model_name: str = "Qwen/Qwen1.5-1.8B",
                 embedding_dim: int = 50):
        """
        Args:
            stage1_model: Pre-trained CLDStage1 model (will be frozen)
            llm_model_name: Name of LLM model to load
            embedding_dim: Dimension of Stage-1 embeddings
        """
        super(CLDStage2, self).__init__()
        
        # Stage-1 model (frozen)
        self.stage1_model = stage1_model
        for param in self.stage1_model.parameters():
            param.requires_grad = False
        self.stage1_model.eval()
        
        # LLM backbone (frozen)
        print(f"Loading LLM model: {llm_model_name}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm.eval()
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get LLM hidden size
        llm_hidden_size = self.llm.config.hidden_size
        
        # Projector (trainable)
        self.projector = Projector(embedding_dim=embedding_dim, llm_hidden_size=llm_hidden_size)
        
        self.embedding_dim = embedding_dim
        self.llm_hidden_size = llm_hidden_size
    
    def get_stage1_embeddings(self, input_seq: torch.Tensor, 
                              seq_len: Optional[torch.Tensor] = None,
                              item_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from Stage-1 model.
        
        Args:
            input_seq: Input sequence tensor of shape (batch_size, max_len)
            seq_len: Actual sequence lengths before padding, shape (batch_size,)
            item_ids: Optional item IDs for which to get embeddings
        
        Returns:
            Dictionary containing:
            - user_emb: User representation (batch_size, embedding_dim)
            - item_embs: Item embeddings (num_items or batch_size, embedding_dim)
        """
        with torch.no_grad():
            # Get user representation
            user_emb = self.stage1_model.get_user_representation(input_seq, seq_len)
            
            # Get item embeddings
            if item_ids is not None:
                # Get embeddings for specific items
                item_embs = self.stage1_model.item_embedding(item_ids)
            else:
                # Get all item embeddings
                all_items = torch.arange(1, self.stage1_model.num_items, device=input_seq.device)
                item_embs = self.stage1_model.item_embedding(all_items)
        
        return {
            'user_emb': user_emb,
            'item_embs': item_embs
        }
    
    def build_prompt_embeddings(self, 
                                 text_token_ids: torch.Tensor,
                                 soft_token_positions: List[List[Tuple[int, int]]],
                                 soft_token_embeddings: List[List[torch.Tensor]],
                                 device: torch.device) -> torch.Tensor:
        """
        Build mixed embeddings by combining text token embeddings and soft token embeddings.
        
        This method replaces text tokens at specified positions with projected soft tokens.
        
        Args:
            text_token_ids: Token IDs tensor (batch_size, seq_len)
            soft_token_positions: List of (start_pos, end_pos) tuples for each sample
            soft_token_embeddings: List of lists of soft token embeddings for each sample
            device: Device to create tensors on
        
        Returns:
            Mixed embeddings tensor of shape (batch_size, seq_len, llm_hidden_size)
        """
        batch_size = text_token_ids.size(0)
        seq_len = text_token_ids.size(1)
        
        # Get LLM token embeddings for all tokens
        llm_embedding_layer = self.llm.get_input_embeddings()
        text_embs = llm_embedding_layer(text_token_ids)  # (batch_size, seq_len, llm_hidden_size)
        
        # Clone to create mixed embeddings
        mixed_embs = text_embs.clone()
        
        # Replace text embeddings with projected soft tokens at specified positions
        for i in range(batch_size):
            soft_positions = soft_token_positions[i]
            soft_embs = soft_token_embeddings[i]
            
            soft_idx = 0
            for pos_start, pos_end in soft_positions:
                if soft_idx < len(soft_embs) and pos_start < seq_len:
                    # Project soft token embedding
                    soft_emb = soft_embs[soft_idx]  # (embedding_dim,)
                    if soft_emb.dim() == 1:
                        soft_emb = soft_emb.unsqueeze(0)
                    projected_soft_emb = self.projector(soft_emb)  # (1, llm_hidden_size)
                    
                    # Replace text embedding at this position
                    pos = min(pos_start, seq_len - 1)
                    mixed_embs[i, pos:pos+1] = projected_soft_emb
                    soft_idx += 1
        
        return mixed_embs
    
    def forward(self, text_token_ids: torch.Tensor, 
                soft_token_positions: List[List[Tuple[int, int]]],
                soft_token_embeddings: List[List[torch.Tensor]],
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, 
                label_start_positions: Optional[List[int]] = None):
        """
        Forward pass through LLM with mixed embeddings.
        
        Args:
            text_token_ids: Token IDs tensor (batch_size, seq_len)
            soft_token_positions: List of soft token positions for each sample
            soft_token_embeddings: List of soft token embeddings for each sample
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Labels for computing loss (batch_size, seq_len)
            label_start_positions: List of positions where labels start (for each sample)
        
        Returns:
            Dictionary with loss and logits
        """
        device = text_token_ids.device
        
        # Build mixed embeddings
        inputs_embeds = self.build_prompt_embeddings(
            text_token_ids, soft_token_positions, soft_token_embeddings, device
        )
        
        # Forward through LLM (without labels first, we'll compute custom loss)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Compute loss only on target part (after "The recommendation is...")
        if labels is not None and label_start_positions is not None:
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            
            # Extract logits and labels only for the target part
            batch_losses = []
            for i, start_pos in enumerate(label_start_positions):
                if start_pos < logits.size(1):
                    # Get logits from start_pos-1 (predict next token)
                    # Get labels from start_pos onwards
                    sample_logits = logits[i, start_pos-1:-1, :]  # (target_len, vocab_size)
                    sample_labels = labels[i, start_pos:]  # (target_len,)
                    
                    # Filter out padding tokens (-100)
                    valid_mask = sample_labels != -100
                    if valid_mask.any():
                        valid_logits = sample_logits[valid_mask]
                        valid_labels = sample_labels[valid_mask]
                        sample_loss = F.cross_entropy(valid_logits, valid_labels)
                        batch_losses.append(sample_loss)
                    else:
                        batch_losses.append(torch.tensor(0.0, device=logits.device, requires_grad=True))
                else:
                    batch_losses.append(torch.tensor(0.0, device=logits.device, requires_grad=True))
            
            if len(batch_losses) > 0:
                loss = torch.stack(batch_losses).mean()
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            outputs.loss = loss
        
        return outputs
    
    def generate(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 50, **generation_kwargs):
        """
        Generate recommendations using LLM.
        
        Args:
            inputs_embeds: Mixed embeddings tensor
            attention_mask: Attention mask
            max_new_tokens: Maximum number of tokens to generate
            **generation_kwargs: Additional generation arguments
        
        Returns:
            Generated token IDs
        """
        # For generation, we need to use the model's generate method
        # But since we're using inputs_embeds, we need a workaround
        # One approach: create dummy input_ids and replace embeddings
        
        batch_size = inputs_embeds.size(0)
        seq_len = inputs_embeds.size(1)
        
        # Create dummy input_ids (will be replaced by inputs_embeds)
        dummy_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=inputs_embeds.device)
        
        # Temporarily replace embedding layer
        original_embedding = self.llm.get_input_embeddings()
        
        class CustomEmbedding(nn.Module):
            def __init__(self, embeds):
                super().__init__()
                self.embeds = embeds
            
            def forward(self, input_ids):
                return self.embeds
        
        custom_emb = CustomEmbedding(inputs_embeds)
        self.llm.set_input_embeddings(custom_emb)
        
        # Generate
        with torch.no_grad():
            generated = self.llm.generate(
                input_ids=dummy_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )
        
        # Restore original embedding
        self.llm.set_input_embeddings(original_embedding)
        
        return generated

