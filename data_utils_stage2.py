import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from data_utils import (
    load_amazon_data, filter_data, create_item_mappings, 
    create_user_sequences, split_sequences
)


class Stage2Dataset(Dataset):
    """
    Dataset class for Stage 2: Returns sequences with item titles.
    """
    
    def __init__(self, sequences: List[List[int]], item_titles: Dict[int, str],
                 max_len: int = 50, num_candidates: int = 10, num_popular: int = 3):
        """
        Args:
            sequences: List of user interaction sequences (list of item IDs)
            item_titles: Dictionary mapping item_idx to title string
            max_len: Maximum sequence length for padding/truncation
            num_candidates: Number of candidate items to include
            num_popular: Number of popular items to highlight
        """
        self.sequences = sequences
        self.item_titles = item_titles
        self.max_len = max_len
        self.num_candidates = num_candidates
        self.num_popular = num_popular
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_seq: Padded/truncated input sequence (max_len)
            target: Target item ID
            seq_len: Actual sequence length before padding
            target_title: Title of target item
            history_items: List of item IDs in history
            candidate_items: List of candidate item IDs
            popular_items: List of popular item IDs
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
        
        # Get target title
        target_title = self.item_titles.get(target, f"Item_{target}")
        
        # History items (non-padded)
        history_items = [item for item in input_seq if item > 0]
        
        # For now, use all items as candidates (will be filtered in collate_fn)
        # In practice, candidates should come from Stage-1 predictions or random sampling
        candidate_items = list(range(1, max(self.item_titles.keys()) + 1))[:self.num_candidates]
        
        # Popular items (simplified: use most frequent items)
        # In practice, should use item popularity scores from Stage-1
        popular_items = candidate_items[:self.num_popular]
        
        return {
            'input_seq': np.array(padded_seq, dtype=np.int64),
            'target': np.int64(target),
            'seq_len': np.int64(seq_len),
            'target_title': target_title,
            'history_items': history_items,
            'candidate_items': candidate_items,
            'popular_items': popular_items
        }


def build_prompt_template(history_items: List[Tuple[str, int]], 
                         candidate_items: List[Tuple[str, int]], 
                         popular_items: List[Tuple[str, int]],
                         target_title: Optional[str] = None) -> Tuple[str, List[int], List[int]]:
    """
    Build prompt template according to the specified format.
    
    Template:
    "[User Representation] is a user representation. This user has interacted with [HISTORY] 
    in the past. Recommend an item for this user from the following set of item titles, 
    [CANDIDATE]. The popular item titles among them are [POPULAR], and try to eliminate 
    the influence of popularity bias as much as possible. The recommendation is..."
    
    Args:
        history_items: List of (title, item_id) tuples for history
        candidate_items: List of (title, item_id) tuples for candidates
        popular_items: List of (title, item_id) tuples for popular items
        target_title: Optional target item title (for training)
    
    Returns:
        prompt_text: Text prompt
        soft_token_positions: List of token positions where soft tokens should be inserted
        soft_token_item_ids: List of item IDs corresponding to each soft token position
    """
    # Build prompt parts
    prompt_parts = []
    soft_token_positions = []  # Token positions (will be filled after tokenization)
    soft_token_item_ids = []  # Item IDs for each soft token
    
    # Part 1: "[User Representation] is a user representation."
    prompt_parts.append("[USER_EMB] is a user representation.")
    soft_token_positions.append(0)  # Will be at position 0 after tokenization
    soft_token_item_ids.append(-1)  # -1 indicates user embedding
    
    # Part 2: "This user has interacted with [HISTORY] in the past."
    if history_items:
        history_titles = [title for title, _ in history_items]
        history_str = ", ".join(history_titles)
        prompt_parts.append(f" This user has interacted with {history_str} in the past.")
        # Each title will be followed by a soft token
        for title, item_id in history_items:
            soft_token_positions.append(None)  # Will be determined after tokenization
            soft_token_item_ids.append(item_id)
    else:
        prompt_parts.append(" This user has no interaction history.")
    
    # Part 3: "Recommend an item for this user from the following set of item titles, [CANDIDATE]."
    if candidate_items:
        candidate_titles = [title for title, _ in candidate_items]
        candidate_str = ", ".join(candidate_titles)
        prompt_parts.append(f" Recommend an item for this user from the following set of item titles, {candidate_str}.")
        # Each title will be followed by a soft token
        for title, item_id in candidate_items:
            soft_token_positions.append(None)
            soft_token_item_ids.append(item_id)
    
    # Part 4: "The popular item titles among them are [POPULAR], and try to eliminate..."
    if popular_items:
        popular_titles = [title for title, _ in popular_items]
        popular_str = ", ".join(popular_titles)
        prompt_parts.append(f" The popular item titles among them are {popular_str}, and try to eliminate the influence of popularity bias as much as possible.")
        # Each title will be followed by a soft token
        for title, item_id in popular_items:
            soft_token_positions.append(None)
            soft_token_item_ids.append(item_id)
    
    # Part 5: "The recommendation is..."
    prompt_parts.append(" The recommendation is")
    
    # Part 6: Target title (training only)
    if target_title is not None:
        prompt_parts.append(f" {target_title}")
    
    # Combine all parts
    prompt_text = "".join(prompt_parts)
    
    return prompt_text, soft_token_positions, soft_token_item_ids


def collate_fn_stage2(batch: List[Dict], tokenizer, stage1_model, 
                      item_titles: Dict[int, str], device: torch.device) -> Dict:
    """
    Collate function for Stage 2: Builds mixed embeddings from text and Stage-1 embeddings.
    
    Args:
        batch: List of samples from Stage2Dataset
        tokenizer: LLM tokenizer
        stage1_model: Frozen Stage-1 model
        item_titles: Dictionary mapping item_idx to title
        device: Device to create tensors on
    
    Returns:
        Dictionary containing:
        - inputs_embeds: Mixed embeddings (batch_size, seq_len, llm_hidden_size)
        - attention_mask: Attention mask (batch_size, seq_len)
        - labels: Labels for loss computation (batch_size, seq_len)
        - label_start_positions: Positions where labels start
    """
    batch_size = len(batch)
    
    # Extract data from batch
    input_seqs = torch.stack([torch.tensor(sample['input_seq']) for sample in batch]).to(device)
    targets = torch.tensor([sample['target'] for sample in batch]).to(device)
    seq_lens = torch.tensor([sample['seq_len'] for sample in batch]).to(device)
    
    # Get Stage-1 embeddings
    with torch.no_grad():
        stage1_embeddings = stage1_model.get_stage1_embeddings(input_seqs, seq_lens)
        user_embs = stage1_embeddings['user_emb']  # (batch_size, embedding_dim)
        all_item_embs = stage1_embeddings['item_embs']  # (num_items-1, embedding_dim)
    
    # Build prompts and collect soft token embeddings
    text_token_ids_list = []
    soft_token_positions_list = []
    soft_token_embeddings_list = []
    labels_list = []
    label_start_positions = []
    
    for i, sample in enumerate(batch):
        history_items = [(item_titles.get(item_id, f"Item_{item_id}"), item_id) 
                         for item_id in sample['history_items'] if item_id > 0]
        candidate_items = [(item_titles.get(item_id, f"Item_{item_id}"), item_id) 
                          for item_id in sample['candidate_items']]
        popular_items = [(item_titles.get(item_id, f"Item_{item_id}"), item_id) 
                        for item_id in sample['popular_items']]
        target_title = sample['target_title']
        
        # Build prompt text
        prompt_text, soft_pos_template, soft_item_ids = build_prompt_template(
            history_items=history_items,
            candidate_items=candidate_items,
            popular_items=popular_items,
            target_title=target_title
        )
        
        # Tokenize prompt
        tokenized = tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=True, max_length=2048)
        token_ids = tokenized['input_ids'][0].tolist()
        
        # Find soft token positions by locating titles in tokenized sequence
        soft_token_positions = []
        soft_token_embeddings = []
        
        # 1. User embedding (at "[USER_EMB]" position)
        # Find "[USER_EMB]" token position
        user_emb_text = "[USER_EMB]"
        user_tokens = tokenizer.encode(user_emb_text, add_special_tokens=False)
        if user_tokens:
            # Find position after user_emb text
            for j in range(len(token_ids) - len(user_tokens) + 1):
                if token_ids[j:j+len(user_tokens)] == user_tokens:
                    soft_token_positions.append(j + len(user_tokens))
                    soft_token_embeddings.append(user_embs[i])
                    break
        
        # 2. History item embeddings (after each title)
        for title, item_id in history_items:
            title_tokens = tokenizer.encode(title, add_special_tokens=False)
            if title_tokens and item_id > 0 and item_id < len(all_item_embs) + 1:
                # Find title position in token_ids
                for j in range(len(token_ids) - len(title_tokens) + 1):
                    if token_ids[j:j+len(title_tokens)] == title_tokens:
                        # Insert soft token after title
                        soft_token_positions.append(j + len(title_tokens))
                        item_emb = all_item_embs[item_id - 1]
                        soft_token_embeddings.append(item_emb)
                        break
        
        # 3. Candidate item embeddings
        for title, item_id in candidate_items:
            title_tokens = tokenizer.encode(title, add_special_tokens=False)
            if title_tokens and item_id > 0 and item_id < len(all_item_embs) + 1:
                for j in range(len(token_ids) - len(title_tokens) + 1):
                    if token_ids[j:j+len(title_tokens)] == title_tokens:
                        soft_token_positions.append(j + len(title_tokens))
                        item_emb = all_item_embs[item_id - 1]
                        soft_token_embeddings.append(item_emb)
                        break
        
        # 4. Popular item embeddings
        for title, item_id in popular_items:
            title_tokens = tokenizer.encode(title, add_special_tokens=False)
            if title_tokens and item_id > 0 and item_id < len(all_item_embs) + 1:
                for j in range(len(token_ids) - len(title_tokens) + 1):
                    if token_ids[j:j+len(title_tokens)] == title_tokens:
                        soft_token_positions.append(j + len(title_tokens))
                        item_emb = all_item_embs[item_id - 1]
                        soft_token_embeddings.append(item_emb)
                        break
        
        # Sort positions and corresponding embeddings
        sorted_indices = sorted(range(len(soft_token_positions)), key=lambda k: soft_token_positions[k])
        soft_token_positions = [soft_token_positions[i] for i in sorted_indices]
        soft_token_embeddings = [soft_token_embeddings[i] for i in sorted_indices]
        
        # Convert to tuples for compatibility
        soft_positions_tuples = [(pos, pos + 1) for pos in soft_token_positions]
        
        # Find position where "The recommendation is" starts
        recommendation_text = "The recommendation is"
        recommendation_tokens = tokenizer.encode(recommendation_text, add_special_tokens=False)
        
        label_start_pos = len(token_ids)  # Default to end
        for j in range(len(token_ids) - len(recommendation_tokens) + 1):
            if token_ids[j:j+len(recommendation_tokens)] == recommendation_tokens:
                label_start_pos = j + len(recommendation_tokens)
                break
        
        # Create labels (only for target part, -100 for rest)
        labels = [-100] * len(token_ids)
        if target_title:
            target_tokens = tokenizer.encode(target_title, add_special_tokens=False)
            if label_start_pos < len(token_ids):
                for j, token_id in enumerate(target_tokens):
                    if label_start_pos + j < len(token_ids):
                        labels[label_start_pos + j] = token_id
        
        text_token_ids_list.append(token_ids)
        soft_token_positions_list.append(soft_positions_tuples)
        soft_token_embeddings_list.append(soft_token_embeddings)
        labels_list.append(labels)
        label_start_positions.append(label_start_pos)
    
    # Build mixed embeddings using Stage-2 model's method
    # For now, we'll create a simplified version here
    # The actual mixing will be done in the model's forward pass
    
    # Pad sequences
    max_len = max(len(ids) for ids in text_token_ids_list)
    padded_token_ids = []
    padded_labels = []
    padded_attention_mask = []
    
    for token_ids, labels in zip(text_token_ids_list, labels_list):
        pad_length = max_len - len(token_ids)
        padded_token_ids.append(token_ids + [tokenizer.pad_token_id] * pad_length)
        padded_labels.append(labels + [-100] * pad_length)
        padded_attention_mask.append([1] * len(token_ids) + [0] * pad_length)
    
    # Convert soft_token_embeddings to list of lists of tensors (keep on CPU, will move to device in training)
    soft_token_embeddings_tensors = []
    for sample_embs in soft_token_embeddings_list:
        sample_tensors = []
        for emb in sample_embs:
            if isinstance(emb, torch.Tensor):
                sample_tensors.append(emb.cpu())
            else:
                sample_tensors.append(torch.tensor(emb, dtype=torch.float32))
        soft_token_embeddings_tensors.append(sample_tensors)
    
    return {
        'text_token_ids': padded_token_ids,
        'soft_token_positions': soft_token_positions_list,
        'soft_token_embeddings': soft_token_embeddings_tensors,
        'labels': torch.tensor(padded_labels, dtype=torch.long, device=device),
        'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long, device=device),
        'label_start_positions': label_start_positions
    }


def get_stage2_loaders(data_path: str, dataset_name: str = 'Games',
                       stage1_model_path: str = None,
                       min_user_interactions: int = 5, min_item_interactions: int = 5,
                       max_len: int = 50, batch_size: int = 4, num_workers: int = 0,
                       num_candidates: int = 10, num_popular: int = 3,
                       device: torch.device = None) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create data loaders for Stage 2 training.
    
    Args:
        data_path: Path to data directory
        dataset_name: Name of dataset
        stage1_model_path: Path to trained Stage-1 model checkpoint
        min_user_interactions: Minimum interactions per user
        min_item_interactions: Minimum interactions per item
        max_len: Maximum sequence length
        batch_size: Batch size (should be small for LLM)
        num_workers: Number of worker processes
        num_candidates: Number of candidate items
        num_popular: Number of popular items
        device: Device to use
    
    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data with metadata
    df = load_amazon_data(data_path, dataset_name, load_metadata=True)
    df_filtered = filter_data(df, min_user_interactions, min_item_interactions)
    
    # Create item mappings
    item_to_idx, idx_to_item = create_item_mappings(df_filtered)
    num_items = len(item_to_idx) + 1
    
    # Extract item titles
    item_titles = {}
    for item_id, idx in item_to_idx.items():
        item_data = df_filtered[df_filtered['item_id'] == item_id]
        if len(item_data) > 0:
            # Try to get title
            title = None
            for col in ['title', 'description', 'summary', 'brand', 'category']:
                if col in item_data.columns:
                    title = item_data[col].iloc[0]
                    if pd.notna(title) and str(title).strip():
                        break
            if title is None or not str(title).strip():
                title = f"Item_{item_id}"
            item_titles[idx] = str(title)[:100]  # Limit length
        else:
            item_titles[idx] = f"Item_{item_id}"
    
    # Create sequences
    sequences = create_user_sequences(df_filtered, item_to_idx)
    
    # Split sequences
    train_sequences, val_sequences, test_sequences = split_sequences(sequences)
    
    # Load Stage-1 model
    if stage1_model_path and os.path.exists(stage1_model_path):
        from models.cld_stage1 import CLDStage1
        checkpoint = torch.load(stage1_model_path, map_location=device)
        stage1_model = CLDStage1(**checkpoint['args'])
        stage1_model.load_state_dict(checkpoint['model_state_dict'])
        stage1_model = stage1_model.to(device)
        stage1_model.eval()
    else:
        raise ValueError(f"Stage-1 model not found at {stage1_model_path}")
    
    # Create datasets
    train_dataset = Stage2Dataset(train_sequences, item_titles, max_len, num_candidates, num_popular)
    val_dataset = Stage2Dataset(val_sequences, item_titles, max_len, num_candidates, num_popular)
    test_dataset = Stage2Dataset(test_sequences, item_titles, max_len, num_candidates, num_popular)
    
    # Create tokenizer (will be loaded with model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn_stage2(
            batch, tokenizer, stage1_model, item_titles, device
        )
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn_stage2(
            batch, tokenizer, stage1_model, item_titles, device
        )
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn_stage2(
            batch, tokenizer, stage1_model, item_titles, device
        )
    )
    
    metadata = {
        'item_to_idx': item_to_idx,
        'idx_to_item': idx_to_item,
        'num_items': num_items,
        'num_users': len(sequences),
        'item_titles': item_titles
    }
    
    return train_loader, val_loader, test_loader, metadata

