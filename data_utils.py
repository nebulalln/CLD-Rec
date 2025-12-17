import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import torch

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("Warning: sentence-transformers not available. Text features will not be extracted.")


class AmazonDataset(Dataset):
    """
    Dataset class for Amazon Review data.
    Handles sequential recommendation with Leave-one-out split.
    """
    
    def __init__(self, sequences: List[List[int]], max_len: int = 50,
                 item_text_embeddings: Optional[Dict[int, np.ndarray]] = None):
        """
        Args:
            sequences: List of user interaction sequences (list of item IDs)
            max_len: Maximum sequence length for padding/truncation
            item_text_embeddings: Optional dictionary mapping item_idx to text embeddings
        """
        self.sequences = sequences
        self.max_len = max_len
        self.item_text_embeddings = item_text_embeddings
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_seq: Padded/truncated input sequence (max_len)
            target: Target item ID
            seq_len: Actual sequence length before padding
            target_text_emb: Optional text embedding for target item
        """
        seq = self.sequences[idx]
        seq_len = len(seq) - 1  # Exclude target item
        
        # Truncate if sequence is too long
        if seq_len > self.max_len:
            input_seq = seq[-self.max_len-1:-1]  # Take last max_len items
            seq_len = self.max_len
        else:
            input_seq = seq[:-1]
        
        # Pad with zeros if sequence is too short
        padded_seq = [0] * (self.max_len - seq_len) + input_seq
        target = seq[-1]
        
        result = {
            'input_seq': np.array(padded_seq, dtype=np.int64),
            'target': np.int64(target),
            'seq_len': np.int64(seq_len)
        }
        
        # Add text embedding if available
        if self.item_text_embeddings is not None and target in self.item_text_embeddings:
            result['target_text_emb'] = self.item_text_embeddings[target].astype(np.float32)
        
        return result


def extract_text_features(df: pd.DataFrame, item_to_idx: Dict[str, int],
                         model_name: str = 'all-MiniLM-L6-v2',
                         batch_size: int = 32) -> Dict[int, np.ndarray]:
    """
    Extract text features for items using SBERT.
    
    Args:
        df: DataFrame with item metadata (should contain 'title' or 'description')
        item_to_idx: Mapping from item ID to integer index
        model_name: Name of SBERT model to use
        batch_size: Batch size for encoding
    
    Returns:
        Dictionary mapping item_idx to text embeddings (numpy array of shape (768,))
    """
    if not SBERT_AVAILABLE:
        print("Warning: sentence-transformers not available. Returning empty text embeddings.")
        return {}
    
    print(f"Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Get unique items and their text
    item_texts = {}
    for item_id, idx in item_to_idx.items():
        # Try to get text from metadata
        item_data = df[df['item_id'] == item_id]
        if len(item_data) > 0:
            # Try 'title' first, then 'description', then 'summary'
            text = None
            for col in ['title', 'description', 'summary', 'brand', 'category']:
                if col in item_data.columns:
                    text = item_data[col].iloc[0]
                    if pd.notna(text) and str(text).strip():
                        break
            
            if text is None or not str(text).strip():
                # Use item_id as fallback
                text = str(item_id)
            
            item_texts[idx] = str(text)
        else:
            # Fallback to item_id
            item_texts[idx] = str(item_id)
    
    # Extract embeddings in batches
    print(f"Extracting text embeddings for {len(item_texts)} items...")
    item_indices = list(item_texts.keys())
    texts = [item_texts[idx] for idx in item_indices]
    
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                             convert_to_numpy=True)
    
    # Create dictionary mapping
    item_embeddings = {idx: emb for idx, emb in zip(item_indices, embeddings)}
    
    print(f"Extracted text embeddings of shape {embeddings.shape[1]} for {len(item_embeddings)} items")
    
    return item_embeddings


def load_amazon_data(data_path: str, dataset_name: str = 'Games', 
                    load_metadata: bool = False) -> pd.DataFrame:
    """
    Load Amazon Review dataset.
    
    Supports both JSON and CSV formats.
    Expected columns: ['user_id', 'item_id', 'timestamp', 'rating', ...]
    
    Args:
        data_path: Path to the data directory
        dataset_name: Name of the dataset (Games, Toys, or Sports)
    
    Returns:
        DataFrame with columns: user_id, item_id, timestamp, rating
    """
    # Try JSON format first
    json_path = os.path.join(data_path, f'{dataset_name}.json')
    csv_path = os.path.join(data_path, f'{dataset_name}.csv')
    
    if os.path.exists(json_path):
        print(f"Loading JSON data from {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        df = pd.DataFrame(data)
    elif os.path.exists(csv_path):
        print(f"Loading CSV data from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Data file not found. Expected {json_path} or {csv_path}"
        )
    
    # Standardize column names
    column_mapping = {
        'reviewerID': 'user_id',
        'asin': 'item_id',
        'unixReviewTime': 'timestamp',
        'reviewTime': 'timestamp',
        'overall': 'rating'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    # Ensure required columns exist
    required_cols = ['user_id', 'item_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Add timestamp if missing (use row index as proxy)
    if 'timestamp' not in df.columns:
        df['timestamp'] = range(len(df))
        print("Warning: timestamp column not found, using row index as proxy")
    
    # Add rating if missing (default to 1)
    if 'rating' not in df.columns:
        df['rating'] = 1
        print("Warning: rating column not found, defaulting to 1")
    
    if load_metadata:
        # Return all columns including metadata (title, description, etc.)
        return df
    else:
        return df[['user_id', 'item_id', 'timestamp', 'rating']]


def filter_data(df: pd.DataFrame, min_user_interactions: int = 5, 
                min_item_interactions: int = 5) -> pd.DataFrame:
    """
    Filter out users and items with fewer than threshold interactions.
    
    Args:
        df: Input DataFrame
        min_user_interactions: Minimum number of interactions per user
        min_item_interactions: Minimum number of interactions per item
    
    Returns:
        Filtered DataFrame
    """
    print(f"Original data: {len(df)} interactions, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
    
    # Filter users
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_user_interactions].index
    df_filtered = df[df['user_id'].isin(valid_users)].copy()
    print(f"After user filtering: {len(df_filtered)} interactions, {df_filtered['user_id'].nunique()} users")
    
    # Filter items
    item_counts = df_filtered['item_id'].value_counts()
    valid_items = item_counts[item_counts >= min_item_interactions].index
    df_filtered = df_filtered[df_filtered['item_id'].isin(valid_items)].copy()
    print(f"After item filtering: {len(df_filtered)} interactions, {df_filtered['item_id'].nunique()} items")
    
    # Re-filter users (some users may have lost items)
    user_counts = df_filtered['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_user_interactions].index
    df_filtered = df_filtered[df_filtered['user_id'].isin(valid_users)].copy()
    print(f"Final data: {len(df_filtered)} interactions, {df_filtered['user_id'].nunique()} users, {df_filtered['item_id'].nunique()} items")
    
    return df_filtered


def create_item_mappings(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create mappings between original item IDs and sequential integer IDs.
    
    Args:
        df: DataFrame with item_id column
    
    Returns:
        item_to_idx: Mapping from original item ID to integer index
        idx_to_item: Mapping from integer index to original item ID
    """
    unique_items = sorted(df['item_id'].unique())
    # Reserve 0 for padding
    item_to_idx = {item: idx + 1 for idx, item in enumerate(unique_items)}
    idx_to_item = {idx + 1: item for item, idx in item_to_idx.items()}
    idx_to_item[0] = '<PAD>'
    
    return item_to_idx, idx_to_item


def create_user_sequences(df: pd.DataFrame, item_to_idx: Dict[str, int]) -> List[List[int]]:
    """
    Create user interaction sequences sorted by timestamp.
    
    Args:
        df: DataFrame with user_id, item_id, timestamp
        item_to_idx: Mapping from item ID to integer index
    
    Returns:
        List of sequences, where each sequence is a list of item indices
    """
    # Sort by user and timestamp
    df_sorted = df.sort_values(['user_id', 'timestamp']).copy()
    
    # Group by user and create sequences
    sequences = []
    for user_id, group in df_sorted.groupby('user_id'):
        item_sequence = [item_to_idx[item_id] for item_id in group['item_id'].values]
        sequences.append(item_sequence)
    
    return sequences


def split_sequences(sequences: List[List[int]]) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    Split sequences using Leave-one-out strategy:
    - Last item: Test
    - Second-to-last item: Validation
    - All others: Training
    
    Args:
        sequences: List of user sequences
    
    Returns:
        train_sequences, val_sequences, test_sequences
    """
    train_sequences = []
    val_sequences = []
    test_sequences = []
    
    for seq in sequences:
        if len(seq) < 3:
            # Skip sequences with fewer than 3 items
            continue
        
        # Training: all items except last two
        train_seq = seq[:-2]
        if len(train_seq) > 0:
            train_sequences.append(train_seq)
        
        # Validation: all items except last one (target is second-to-last)
        val_seq = seq[:-1]
        val_sequences.append(val_seq)
        
        # Test: full sequence (target is last item)
        test_sequences.append(seq)
    
    print(f"Split sequences: {len(train_sequences)} train, {len(val_sequences)} val, {len(test_sequences)} test")
    
    return train_sequences, val_sequences, test_sequences


def get_loaders(data_path: str, dataset_name: str = 'Games', 
                min_user_interactions: int = 5, min_item_interactions: int = 5,
                max_len: int = 50, batch_size: int = 128,
                num_workers: int = 4, extract_text_features: bool = True,
                sbert_model: str = 'all-MiniLM-L6-v2') -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Main function to create data loaders for training, validation, and testing.
    
    Args:
        data_path: Path to data directory
        dataset_name: Name of dataset (Games, Toys, or Sports)
        min_user_interactions: Minimum interactions per user
        min_item_interactions: Minimum interactions per item
        max_len: Maximum sequence length
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        extract_text_features: Whether to extract text features using SBERT
        sbert_model: Name of SBERT model to use
    
    Returns:
        train_loader, val_loader, test_loader, metadata (containing item mappings and vocab size)
    """
    # Load and filter data (with metadata if text features are needed)
    load_metadata = extract_text_features
    df = load_amazon_data(data_path, dataset_name, load_metadata=load_metadata)
    df_filtered = filter_data(df, min_user_interactions, min_item_interactions)
    
    # Create item mappings
    item_to_idx, idx_to_item = create_item_mappings(df_filtered)
    num_items = len(item_to_idx) + 1  # +1 for padding token
    
    # Extract text features if requested
    item_text_embeddings = None
    if extract_text_features and SBERT_AVAILABLE:
        # Load full data again with metadata for text extraction
        df_full = load_amazon_data(data_path, dataset_name, load_metadata=True)
        df_full_filtered = df_full[df_full['item_id'].isin(df_filtered['item_id'].unique())]
        item_text_embeddings = extract_text_features(df_full_filtered, item_to_idx, 
                                                     model_name=sbert_model)
    elif extract_text_features and not SBERT_AVAILABLE:
        print("Warning: Text feature extraction requested but sentence-transformers not available.")
    
    # Create sequences
    sequences = create_user_sequences(df_filtered, item_to_idx)
    
    # Split sequences
    train_sequences, val_sequences, test_sequences = split_sequences(sequences)
    
    # Create datasets
    train_dataset = AmazonDataset(train_sequences, max_len=max_len, 
                                 item_text_embeddings=item_text_embeddings)
    val_dataset = AmazonDataset(val_sequences, max_len=max_len,
                               item_text_embeddings=item_text_embeddings)
    test_dataset = AmazonDataset(test_sequences, max_len=max_len,
                                item_text_embeddings=item_text_embeddings)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    metadata = {
        'item_to_idx': item_to_idx,
        'idx_to_item': idx_to_item,
        'num_items': num_items,
        'num_users': len(sequences),
        'item_text_embeddings': item_text_embeddings,
        'text_embedding_dim': 768 if item_text_embeddings else None
    }
    
    return train_loader, val_loader, test_loader, metadata

