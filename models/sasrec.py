import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PointWiseFeedForward(nn.Module):
    """
    Point-wise feed-forward network with two linear layers and dropout.
    """
    def __init__(self, hidden_size: int, dropout: float = 0.2):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
        Returns:
            Output tensor of same shape
        """
        x = x.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
        x = self.dropout1(F.relu(self.conv1(x)))
        x = self.dropout2(self.conv2(x))
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_size)
        return x


class SASRecBlock(nn.Module):
    """
    Single transformer block with self-attention and feed-forward network.
    """
    def __init__(self, hidden_size: int, num_heads: int = 1, dropout: float = 0.2):
        super(SASRecBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = PointWiseFeedForward(hidden_size, dropout)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            attn_mask: Attention mask (batch_size, seq_len, seq_len)
        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = self.ln2(x + ffn_output)
        
        return x


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation Model.
    
    Architecture:
    1. Item Embedding Layer
    2. Positional Embedding Layer
    3. Stack of Self-Attention Blocks
    4. Output Layer
    """
    
    def __init__(self, num_items: int, hidden_size: int = 50, num_blocks: int = 2,
                 num_heads: int = 1, dropout: float = 0.2, max_len: int = 50):
        """
        Args:
            num_items: Number of items (vocab size, including padding token)
            hidden_size: Hidden dimension size (default 50 as per paper)
            num_blocks: Number of self-attention blocks
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super(SASRec, self).__init__()
        
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_len = max_len
        
        # Item Embedding Layer
        self.item_embedding = nn.Embedding(num_items, hidden_size, padding_idx=0)
        
        # Positional Embedding Layer
        self.pos_embedding = nn.Embedding(max_len, hidden_size)
        
        # Self-Attention Blocks
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_size, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        self.ln = nn.LayerNorm(hidden_size)
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
    
    def create_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal attention mask (lower triangular matrix).
        Prevents attention to future positions.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
        
        Returns:
            Attention mask of shape (seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        return mask
    
    def forward(self, input_seq: torch.Tensor, seq_len: torch.Tensor = None):
        """
        Forward pass through SASRec model.
        
        Args:
            input_seq: Input sequence tensor of shape (batch_size, max_len)
            seq_len: Actual sequence lengths before padding, shape (batch_size,)
        
        Returns:
            Sequence representations of shape (batch_size, max_len, hidden_size)
        """
        batch_size, seq_length = input_seq.shape
        
        # Create positional indices
        positions = torch.arange(seq_length, device=input_seq.device).unsqueeze(0).expand(batch_size, -1)
        
        # Item embeddings
        item_emb = self.item_embedding(input_seq)  # (batch_size, seq_len, hidden_size)
        
        # Positional embeddings
        pos_emb = self.pos_embedding(positions)  # (batch_size, seq_len, hidden_size)
        
        # Combine embeddings
        x = item_emb + pos_emb
        x = self.dropout(x)
        
        # Create attention mask (causal mask)
        attn_mask = self.create_attention_mask(seq_length, input_seq.device)
        
        # Pass through self-attention blocks
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        
        # Final layer normalization
        x = self.ln(x)
        
        return x
    
    def predict(self, input_seq: torch.Tensor, seq_len: torch.Tensor = None):
        """
        Predict next item given input sequence.
        Returns the representation of the last position in the sequence.
        
        Args:
            input_seq: Input sequence tensor of shape (batch_size, max_len)
            seq_len: Actual sequence lengths before padding, shape (batch_size,)
        
        Returns:
            Last position representation of shape (batch_size, hidden_size)
        """
        seq_repr = self.forward(input_seq, seq_len)
        
        # Get the last valid position for each sequence
        if seq_len is not None:
            # Use actual sequence lengths
            batch_indices = torch.arange(seq_repr.size(0), device=seq_repr.device)
            last_positions = seq_len - 1  # Last valid position (0-indexed)
            last_repr = seq_repr[batch_indices, last_positions]
        else:
            # Use the last position
            last_repr = seq_repr[:, -1, :]
        
        return last_repr
    
    def compute_loss(self, input_seq: torch.Tensor, target: torch.Tensor, 
                     seq_len: torch.Tensor = None, item_embeddings: torch.Tensor = None):
        """
        Compute cross-entropy loss for next item prediction.
        
        Args:
            input_seq: Input sequence tensor of shape (batch_size, max_len)
            target: Target item indices of shape (batch_size,) - should not contain padding (0)
            seq_len: Actual sequence lengths before padding, shape (batch_size,)
            item_embeddings: Optional pre-computed item embeddings (num_items, hidden_size)
                           If None, uses self.item_embedding
        
        Returns:
            Loss value (scalar tensor)
        """
        # Get sequence representation
        last_repr = self.predict(input_seq, seq_len)  # (batch_size, hidden_size)
        
        # Get item embeddings for all items (excluding padding token 0)
        if item_embeddings is None:
            all_items = torch.arange(1, self.num_items, device=input_seq.device)  # Exclude padding
            item_emb = self.item_embedding(all_items)  # (num_items-1, hidden_size)
        else:
            item_emb = item_embeddings[1:]  # Exclude padding
        
        # Compute logits: dot product between last_repr and all item embeddings
        logits = torch.matmul(last_repr, item_emb.t())  # (batch_size, num_items-1)
        
        # Adjust target indices (subtract 1 to exclude padding token)
        # Target should be >= 1 (no padding), so target_adjusted >= 0
        target_adjusted = target - 1
        
        # Ensure targets are valid (in case of any edge cases)
        valid_mask = (target > 0) & (target < self.num_items)
        if not valid_mask.all():
            # Filter out invalid targets
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                return torch.tensor(0.0, device=input_seq.device, requires_grad=True)
            last_repr = last_repr[valid_indices]
            logits = torch.matmul(last_repr, item_emb.t())
            target_adjusted = target_adjusted[valid_indices]
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, target_adjusted)
        
        return loss

