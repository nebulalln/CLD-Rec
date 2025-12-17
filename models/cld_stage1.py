import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .sasrec import SASRec


class UserConformityModule(nn.Module):
    """
    User Module (Yu): Predicts user conformity score.
    Input: User representation (aggregated from sequence)
    Output: User conformity score y_u
    """
    def __init__(self, hidden_size: int = 50, dropout: float = 0.2):
        """
        Args:
            hidden_size: Hidden dimension size
            dropout: Dropout rate
        """
        super(UserConformityModule, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # Output single conformity score
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, user_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_repr: User representation tensor of shape (batch_size, hidden_size)
        
        Returns:
            User conformity scores of shape (batch_size, 1)
        """
        y_u = self.mlp(user_repr)
        return y_u.squeeze(-1)  # (batch_size,)


class ItemPopularityModule(nn.Module):
    """
    Item Module (Yi): Predicts item popularity score.
    Input: Item embedding
    Output: Item popularity score y_i
    """
    def __init__(self, hidden_size: int = 50, dropout: float = 0.2):
        """
        Args:
            hidden_size: Hidden dimension size
            dropout: Dropout rate
        """
        super(ItemPopularityModule, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # Output single popularity score
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, item_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_emb: Item embedding tensor of shape (batch_size, hidden_size) or (num_items, hidden_size)
        
        Returns:
            Item popularity scores of shape (batch_size,) or (num_items,)
        """
        y_i = self.mlp(item_emb)
        return y_i.squeeze(-1)  # (batch_size,) or (num_items,)


class SemanticAlignmentModule(nn.Module):
    """
    Semantic Alignment Module for aligning item embeddings with text embeddings.
    
    Contains two encoders:
    - Item Encoder (f_I): Maps item embedding (hidden_size) to alignment space
    - Text Encoder (f_T): Maps SBERT embedding (768) to alignment space
    """
    def __init__(self, item_embedding_dim: int = 50, text_embedding_dim: int = 768,
                 alignment_dim: int = 50, dropout: float = 0.2):
        """
        Args:
            item_embedding_dim: Dimension of item embeddings from SASRec (default: 50)
            text_embedding_dim: Dimension of SBERT text embeddings (default: 768)
            alignment_dim: Dimension of aligned space (default: 50)
            dropout: Dropout rate
        """
        super(SemanticAlignmentModule, self).__init__()
        
        # Item Encoder (f_I): Maps item embedding to alignment space
        self.item_encoder = nn.Sequential(
            nn.Linear(item_embedding_dim, alignment_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(alignment_dim)
        )
        
        # Text Encoder (f_T): Maps text embedding to alignment space
        self.text_encoder = nn.Sequential(
            nn.Linear(text_embedding_dim, alignment_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(alignment_dim * 2, alignment_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(alignment_dim)
        )
        
        self.alignment_dim = alignment_dim
        self._init_weights()
    
    def _init_weights(self):
        """Initialize encoder weights."""
        for module in [self.item_encoder, self.text_encoder]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def encode_item(self, item_emb: torch.Tensor) -> torch.Tensor:
        """
        Encode item embedding to alignment space.
        
        Args:
            item_emb: Item embedding tensor of shape (..., item_embedding_dim)
        
        Returns:
            Aligned item representation of shape (..., alignment_dim)
        """
        return self.item_encoder(item_emb)
    
    def encode_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Encode text embedding to alignment space.
        
        Args:
            text_emb: Text embedding tensor of shape (..., text_embedding_dim)
        
        Returns:
            Aligned text representation of shape (..., alignment_dim)
        """
        return self.text_encoder(text_emb)
    
    def forward(self, item_emb: torch.Tensor, text_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode both item and text embeddings to alignment space.
        
        Args:
            item_emb: Item embedding tensor of shape (..., item_embedding_dim)
            text_emb: Text embedding tensor of shape (..., text_embedding_dim)
        
        Returns:
            Tuple of (aligned_item_emb, aligned_text_emb)
        """
        aligned_item = self.encode_item(item_emb)
        aligned_text = self.encode_text(text_emb)
        return aligned_item, aligned_text


class CLDStage1(nn.Module):
    """
    CLD-Rec Stage 1: Causal-Enhanced Debiased Recommender System
    
    This stage implements causal intervention to eliminate:
    - User Conformity Bias (U -> Y)
    - Item Popularity Bias (I -> Y)
    
    The true matching signal (U & I -> K -> Y) is preserved through the SASRec backbone.
    """
    
    def __init__(self, num_items: int, hidden_size: int = 50, num_blocks: int = 2,
                 num_heads: int = 1, dropout: float = 0.2, max_len: int = 50,
                 text_embedding_dim: int = 768, use_semantic_alignment: bool = True):
        """
        Args:
            num_items: Number of items (vocab size, including padding token)
            hidden_size: Hidden dimension size (default 50 as per paper)
            num_blocks: Number of self-attention blocks in SASRec
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_len: Maximum sequence length
            text_embedding_dim: Dimension of text embeddings from SBERT (default: 768)
            use_semantic_alignment: Whether to use semantic alignment module
        """
        super(CLDStage1, self).__init__()
        
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.use_semantic_alignment = use_semantic_alignment
        
        # Matching Module (Yk): SASRec backbone
        self.matching_module = SASRec(
            num_items=num_items,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout=dropout,
            max_len=max_len
        )
        
        # User Module (Yu): Predicts user conformity
        self.user_module = UserConformityModule(hidden_size, dropout)
        
        # Item Module (Yi): Predicts item popularity
        self.item_module = ItemPopularityModule(hidden_size, dropout)
        
        # Item embedding (shared with matching module)
        self.item_embedding = self.matching_module.item_embedding
        
        # Semantic Alignment Module
        if use_semantic_alignment:
            self.alignment_module = SemanticAlignmentModule(
                item_embedding_dim=hidden_size,
                text_embedding_dim=text_embedding_dim,
                alignment_dim=hidden_size,
                dropout=dropout
            )
        else:
            self.alignment_module = None
    
    def get_user_representation(self, input_seq: torch.Tensor, 
                                 seq_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get user representation by aggregating sequence representation.
        Uses average pooling over the sequence.
        
        Args:
            input_seq: Input sequence tensor of shape (batch_size, max_len)
            seq_len: Actual sequence lengths before padding, shape (batch_size,)
        
        Returns:
            User representation of shape (batch_size, hidden_size)
        """
        seq_repr = self.matching_module.forward(input_seq, seq_len)  # (batch_size, max_len, hidden_size)
        
        if seq_len is not None:
            # Mask out padding positions and compute average
            batch_size = seq_repr.size(0)
            mask = torch.arange(self.max_len, device=seq_repr.device).unsqueeze(0) < seq_len.unsqueeze(1)
            mask = mask.float().unsqueeze(-1)  # (batch_size, max_len, 1)
            
            masked_repr = seq_repr * mask
            user_repr = masked_repr.sum(dim=1) / seq_len.float().unsqueeze(-1)  # (batch_size, hidden_size)
        else:
            # Simple average pooling
            user_repr = seq_repr.mean(dim=1)  # (batch_size, hidden_size)
        
        return user_repr
    
    def forward(self, input_seq: torch.Tensor, seq_len: Optional[torch.Tensor] = None,
                return_components: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through CLD Stage 1.
        
        Args:
            input_seq: Input sequence tensor of shape (batch_size, max_len)
            seq_len: Actual sequence lengths before padding, shape (batch_size,)
            return_components: If True, returns individual components (y_k, y_u, y_i)
        
        Returns:
            If return_components=False:
                Final logits of shape (batch_size, num_items-1)
            If return_components=True:
                (final_logits, y_k, y_u, y_i)
        """
        batch_size = input_seq.size(0)
        
        # 1. Get matching score (y_k) from SASRec
        user_seq_repr = self.matching_module.predict(input_seq, seq_len)  # (batch_size, hidden_size)
        
        # Get item embeddings for all items (excluding padding)
        all_items = torch.arange(1, self.num_items, device=input_seq.device)
        all_item_emb = self.item_embedding(all_items)  # (num_items-1, hidden_size)
        
        # Compute matching logits: y_k = user_seq_repr @ item_emb^T
        y_k = torch.matmul(user_seq_repr, all_item_emb.t())  # (batch_size, num_items-1)
        
        # 2. Get user conformity score (y_u)
        user_repr = self.get_user_representation(input_seq, seq_len)  # (batch_size, hidden_size)
        y_u = self.user_module(user_repr)  # (batch_size,)
        
        # 3. Get item popularity scores (y_i) for all items
        y_i_all = self.item_module(all_item_emb)  # (num_items-1,)
        
        # 4. Apply causal intervention: y_ui = y_k * sigmoid(y_i) * sigmoid(y_u)
        # Expand y_u and y_i to match y_k dimensions
        y_u_expanded = torch.sigmoid(y_u).unsqueeze(1)  # (batch_size, 1)
        y_i_expanded = torch.sigmoid(y_i_all).unsqueeze(0)  # (1, num_items-1)
        
        # Apply intervention: multiply matching score by debiasing factors
        final_logits = y_k * y_u_expanded * y_i_expanded  # (batch_size, num_items-1)
        
        if return_components:
            return final_logits, y_k, y_u, y_i_all
        else:
            return final_logits
    
    def predict(self, input_seq: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict next item given input sequence.
        
        Args:
            input_seq: Input sequence tensor of shape (batch_size, max_len)
            seq_len: Actual sequence lengths before padding, shape (batch_size,)
        
        Returns:
            Prediction logits of shape (batch_size, num_items-1)
        """
        return self.forward(input_seq, seq_len, return_components=False)
    
    def compute_loss(self, input_seq: torch.Tensor, target: torch.Tensor,
                     seq_len: Optional[torch.Tensor] = None,
                     target_text_emb: Optional[torch.Tensor] = None,
                     alpha: float = 0.1, beta: float = 0.1, gamma: float = 0.1) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss including L_rec, L_U, L_I, and L_matching.
        
        Args:
            input_seq: Input sequence tensor of shape (batch_size, max_len)
            target: Target item indices of shape (batch_size,) - should not contain padding (0)
            seq_len: Actual sequence lengths before padding, shape (batch_size,)
            target_text_emb: Optional text embeddings for target items (batch_size, text_embedding_dim)
            alpha: Weight for user conformity loss L_U
            beta: Weight for item popularity loss L_I
            gamma: Weight for semantic alignment loss L_matching
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary containing individual loss components
        """
        # Get all components
        final_logits, y_k, y_u, y_i_all = self.forward(
            input_seq, seq_len, return_components=True
        )
        
        # 1. L_rec: Recommendation loss (Cross-Entropy on final prediction)
        target_adjusted = target - 1  # Adjust for padding token
        L_rec = F.cross_entropy(final_logits, target_adjusted)
        
        # 2. L_U: User conformity loss
        # User conformity: tendency to interact with popular items
        # Observed conformity = average popularity of items in each user's sequence
        batch_size = input_seq.size(0)
        observed_conformities = []
        
        for i in range(batch_size):
            user_seq = input_seq[i]  # (max_len,)
            if seq_len is not None:
                valid_len = seq_len[i].item()
                user_items = user_seq[:valid_len]  # Get valid items only
            else:
                user_items = user_seq[user_seq > 0]  # Remove padding
            
            if len(user_items) > 0:
                # Get popularity scores for items in this user's sequence
                item_indices = user_items - 1  # Adjust for padding
                item_indices = torch.clamp(item_indices, 0, len(y_i_all) - 1)
                item_popularity = y_i_all[item_indices]  # Popularity scores
                
                # Average popularity = observed conformity for this user
                avg_popularity = torch.sigmoid(item_popularity).mean()
                observed_conformities.append(avg_popularity)
            else:
                # Default conformity if no valid items
                observed_conformities.append(torch.tensor(0.5, device=input_seq.device))
        
        if len(observed_conformities) > 0:
            observed_conformity = torch.stack(observed_conformities)  # (batch_size,)
            predicted_conformity = torch.sigmoid(y_u)  # (batch_size,)
            
            # L_U: MSE between predicted and observed conformity
            L_U = F.mse_loss(predicted_conformity, observed_conformity)
        else:
            L_U = torch.tensor(0.0, device=input_seq.device, requires_grad=True)
        
        # 3. L_I: Item popularity loss
        # Encourage item module to learn popularity patterns
        # Observed popularity: frequency of items in the batch (proxy for global popularity)
        batch_items = input_seq.flatten()  # Flatten all items in batch
        batch_items = batch_items[batch_items > 0]  # Remove padding
        
        if len(batch_items) > 0:
            # Count item frequencies in batch
            item_counts = torch.bincount(batch_items, minlength=self.num_items)
            item_counts = item_counts[1:]  # Exclude padding (shape: num_items-1)
            item_counts = item_counts.float()
            
            # Normalize to probability distribution
            if item_counts.sum() > 0:
                observed_popularity = item_counts / item_counts.sum()
            else:
                observed_popularity = torch.ones(self.num_items - 1, device=input_seq.device) / (self.num_items - 1)
            
            # L_I: KL divergence or MSE between predicted and observed popularity
            # Use softmax to normalize predicted popularity scores
            predicted_popularity = torch.softmax(y_i_all, dim=0)
            
            # Use KL divergence for better probability matching
            # Add small epsilon to avoid log(0)
            epsilon = 1e-8
            predicted_popularity = predicted_popularity + epsilon
            predicted_popularity = predicted_popularity / predicted_popularity.sum()
            observed_popularity = observed_popularity + epsilon
            observed_popularity = observed_popularity / observed_popularity.sum()
            
            L_I = F.kl_div(
                torch.log(predicted_popularity + epsilon),
                observed_popularity,
                reduction='batchmean'
            )
        else:
            L_I = torch.tensor(0.0, device=input_seq.device, requires_grad=True)
        
        # 4. L_matching: Semantic alignment loss
        L_matching = torch.tensor(0.0, device=input_seq.device, requires_grad=True)
        if self.use_semantic_alignment and self.alignment_module is not None and target_text_emb is not None:
            # Get item embeddings for target items
            target_item_emb = self.item_embedding(target)  # (batch_size, hidden_size)
            
            # Encode to alignment space
            aligned_item_emb, aligned_text_emb = self.alignment_module(
                target_item_emb, target_text_emb
            )
            
            # Compute MSE loss between aligned representations
            # Option 1: MSE Loss (simple and effective)
            L_matching = F.mse_loss(aligned_item_emb, aligned_text_emb)
            
            # Option 2: Contrastive Loss (alternative, commented out)
            # # Normalize embeddings
            # aligned_item_emb_norm = F.normalize(aligned_item_emb, p=2, dim=1)
            # aligned_text_emb_norm = F.normalize(aligned_text_emb, p=2, dim=1)
            # # Cosine similarity (positive pairs should be close to 1)
            # similarity = (aligned_item_emb_norm * aligned_text_emb_norm).sum(dim=1)
            # L_matching = F.mse_loss(similarity, torch.ones_like(similarity))
        
        # Total loss: L_rec + alpha * L_U + beta * L_I + gamma * L_matching
        total_loss = L_rec + alpha * L_U + beta * L_I + gamma * L_matching
        
        loss_dict = {
            'L_rec': L_rec.item(),
            'L_U': L_U.item(),
            'L_I': L_I.item(),
            'L_matching': L_matching.item() if isinstance(L_matching, torch.Tensor) else L_matching,
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def get_debiased_collaborative_knowledge(self, input_seq: torch.Tensor,
                                            seq_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract debiased collaborative knowledge (user-item matching scores after intervention).
        This will be used in Stage 2 for LLM integration.
        
        Args:
            input_seq: Input sequence tensor of shape (batch_size, max_len)
            seq_len: Actual sequence lengths before padding, shape (batch_size,)
        
        Returns:
            Debiased matching scores of shape (batch_size, num_items-1)
        """
        return self.predict(input_seq, seq_len)

