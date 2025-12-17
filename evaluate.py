import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict


def compute_ndcg_at_k(predicted_scores: np.ndarray, ground_truth: int, k: int) -> float:
    """
    Compute NDCG@K for a single sample.
    
    Args:
        predicted_scores: Array of predicted scores for all items
        ground_truth: Index of ground truth item
        k: Top-K for evaluation
    
    Returns:
        NDCG@K score
    """
    # Get top-k predicted items
    top_k_indices = np.argsort(predicted_scores)[::-1][:k]
    
    # Check if ground truth is in top-k
    if ground_truth in top_k_indices:
        rank = np.where(top_k_indices == ground_truth)[0][0] + 1
        # DCG@K
        dcg = 1.0 / np.log2(rank + 1)
        # IDCG@K (ideal DCG, since we only have one relevant item)
        idcg = 1.0 / np.log2(2)
        ndcg = dcg / idcg
    else:
        ndcg = 0.0
    
    return ndcg


def compute_hr_at_k(predicted_scores: np.ndarray, ground_truth: int, k: int) -> float:
    """
    Compute Hit Rate@K for a single sample.
    
    Args:
        predicted_scores: Array of predicted scores for all items
        ground_truth: Index of ground truth item
        k: Top-K for evaluation
    
    Returns:
        HR@K score (0 or 1)
    """
    # Get top-k predicted items
    top_k_indices = np.argsort(predicted_scores)[::-1][:k]
    
    # Check if ground truth is in top-k
    hit = 1.0 if ground_truth in top_k_indices else 0.0
    
    return hit


def evaluate_stage1(model, data_loader, device, k_list: List[int] = [5, 10]) -> Dict[str, float]:
    """
    Evaluate Stage-1 model using standard metrics.
    
    Args:
        model: CLDStage1 model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        k_list: List of K values for evaluation
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    all_ndcg = {f'NDCG@{k}': [] for k in k_list}
    all_hr = {f'HR@{k}': [] for k in k_list}
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating Stage-1', leave=False):
            input_seq = batch['input_seq'].to(device)
            target = batch['target'].to(device)
            seq_len = batch['seq_len'].to(device)
            
            # Get predictions
            logits = model.predict(input_seq, seq_len)  # (batch_size, num_items-1)
            
            # Convert to numpy
            logits_np = logits.cpu().numpy()
            targets_np = (target - 1).cpu().numpy()  # Adjust for padding token
            
            # Compute metrics for each sample
            for i in range(len(target)):
                pred_scores = logits_np[i]
                gt = targets_np[i]
                
                for k in k_list:
                    ndcg = compute_ndcg_at_k(pred_scores, gt, k)
                    hr = compute_hr_at_k(pred_scores, gt, k)
                    
                    all_ndcg[f'NDCG@{k}'].append(ndcg)
                    all_hr[f'HR@{k}'].append(hr)
    
    # Average metrics
    metrics = {}
    for k in k_list:
        metrics[f'NDCG@{k}'] = np.mean(all_ndcg[f'NDCG@{k}'])
        metrics[f'HR@{k}'] = np.mean(all_hr[f'HR@{k}'])
    
    return metrics


def evaluate_stage2(model, data_loader, device, tokenizer, k_list: List[int] = [5, 10]) -> Dict[str, float]:
    """
    Evaluate Stage-2 model using standard metrics.
    
    Note: Stage-2 generates text recommendations, so we need to match generated
    item titles with ground truth. This is a simplified version.
    
    Args:
        model: CLDStage2 model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        tokenizer: Tokenizer for decoding
        k_list: List of K values for evaluation
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    # For Stage-2, we'll use the logits from the model to rank items
    # This is a simplified approach - in practice, you might need to decode
    # generated text and match with item titles
    
    all_ndcg = {f'NDCG@{k}': [] for k in k_list}
    all_hr = {f'HR@{k}': [] for k in k_list}
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating Stage-2', leave=False):
            text_token_ids = torch.tensor(batch['text_token_ids'], dtype=torch.long, device=device)
            soft_token_positions = batch['soft_token_positions']
            soft_token_embeddings = batch['soft_token_embeddings']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            # Convert soft_token_embeddings to tensors
            soft_token_embeddings_tensors = []
            for sample_embs in soft_token_embeddings:
                sample_tensors = [emb.to(device) if isinstance(emb, torch.Tensor) else torch.tensor(emb, device=device) 
                                 for emb in sample_embs]
                soft_token_embeddings_tensors.append(sample_tensors)
            
            # Forward pass
            outputs = model(
                text_token_ids=text_token_ids,
                soft_token_positions=soft_token_positions,
                soft_token_embeddings=soft_token_embeddings_tensors,
                attention_mask=attention_mask
            )
            
            # Extract logits for ranking (simplified - use last hidden state)
            # In practice, you might need more sophisticated ranking
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            
            # For now, return placeholder metrics
            # In a full implementation, you would decode recommendations and match with ground truth
            batch_size = logits.size(0)
            for k in k_list:
                all_ndcg[f'NDCG@{k}'].extend([0.0] * batch_size)
                all_hr[f'HR@{k}'].extend([0.0] * batch_size)
    
    # Average metrics
    metrics = {}
    for k in k_list:
        metrics[f'NDCG@{k}'] = np.mean(all_ndcg[f'NDCG@{k}']) if all_ndcg[f'NDCG@{k}'] else 0.0
        metrics[f'HR@{k}'] = np.mean(all_hr[f'HR@{k}']) if all_hr[f'HR@{k}'] else 0.0
    
    return metrics


def create_unbiased_test_set(data_loader, num_negatives: int = 100, 
                             num_items: int = None, 
                             exclude_items: Optional[List[int]] = None,
                             random_seed: int = 42) -> List[Dict]:
    """
    Create an unbiased test set by uniformly sampling negative items.
    
    This function ensures that negative samples are sampled uniformly at random,
    rather than based on popularity, to eliminate popularity bias in evaluation.
    
    Args:
        data_loader: Original test data loader
        num_negatives: Number of negative samples to sample uniformly
        num_items: Total number of items
        exclude_items: List of item indices to exclude from negative sampling
        random_seed: Random seed for reproducibility
    
    Returns:
        List of test samples with uniformly sampled negatives
    """
    np.random.seed(random_seed)
    unbiased_samples = []
    
    # Get all valid item indices (excluding padding and excluded items)
    if exclude_items is None:
        exclude_items = [0]  # Exclude padding
    else:
        exclude_items = list(set(exclude_items + [0]))
    
    valid_items = [i for i in range(1, num_items) if i not in exclude_items]
    
    for batch in data_loader:
        input_seq = batch['input_seq']
        target = batch['target']
        seq_len = batch['seq_len']
        
        batch_size = len(target)
        
        for i in range(batch_size):
            gt_item = target[i].item()
            
            # Exclude ground truth and items in history from negative sampling
            history_items = set(input_seq[i][:seq_len[i].item()].tolist())
            exclude_set = set(exclude_items + [gt_item] + list(history_items))
            
            # Get valid negative items (uniformly sampled, not popularity-based)
            valid_negatives = [item for item in valid_items if item not in exclude_set]
            
            # Uniformly sample negatives (this is the key: uniform sampling, not popularity-based)
            if len(valid_negatives) >= num_negatives:
                sampled_negatives = np.random.choice(valid_negatives, size=num_negatives, replace=False).tolist()
            else:
                # If not enough negatives, use all available and pad with replacement
                sampled_negatives = valid_negatives.copy()
                if len(sampled_negatives) < num_negatives:
                    additional = np.random.choice(valid_negatives, 
                                                 size=num_negatives - len(sampled_negatives), 
                                                 replace=True).tolist()
                    sampled_negatives.extend(additional)
            
            # Shuffle to ensure ground truth position is random in the list
            # But we'll keep track of ground truth position
            test_items = [gt_item] + sampled_negatives
            gt_position = 0  # Ground truth is at position 0
            
            unbiased_samples.append({
                'input_seq': input_seq[i],
                'target': target[i],
                'seq_len': seq_len[i],
                'test_items': test_items,  # Ground truth + uniformly sampled negatives
                'gt_position': gt_position  # Ground truth position
            })
    
    return unbiased_samples


def evaluate_unbiased_stage1(model, unbiased_test_set, device, k_list: List[int] = [5, 10]) -> Dict[str, float]:
    """
    Evaluate Stage-1 model on unbiased test set.
    
    Args:
        model: CLDStage1 model
        unbiased_test_set: List of test samples with uniformly sampled negatives
        device: Device to run evaluation on
        k_list: List of K values for evaluation
    
    Returns:
        Dictionary with unbiased metrics
    """
    model.eval()
    
    all_ndcg = {f'NDCG@{k}': [] for k in k_list}
    all_hr = {f'HR@{k}': [] for k in k_list}
    
    with torch.no_grad():
        for sample in tqdm(unbiased_test_set, desc='Unbiased Evaluation Stage-1', leave=False):
            input_seq = sample['input_seq'].unsqueeze(0).to(device)
            seq_len = sample['seq_len'].unsqueeze(0).to(device)
            test_items = sample['test_items']  # List of item indices (gt + negatives)
            gt_position = sample['gt_position']
            
            # Get predictions for all items
            logits = model.predict(input_seq, seq_len)  # (1, num_items-1)
            
            # Extract scores for test items only
            test_item_scores = []
            for item_idx in test_items:
                if item_idx > 0:  # Exclude padding
                    score_idx = item_idx - 1  # Adjust for padding
                    if score_idx < logits.size(1):
                        test_item_scores.append(logits[0, score_idx].item())
                    else:
                        test_item_scores.append(-float('inf'))
                else:
                    test_item_scores.append(-float('inf'))
            
            test_item_scores = np.array(test_item_scores)
            
            # Compute metrics
            for k in k_list:
                ndcg = compute_ndcg_at_k(test_item_scores, gt_position, k)
                hr = compute_hr_at_k(test_item_scores, gt_position, k)
                
                all_ndcg[f'NDCG@{k}'].append(ndcg)
                all_hr[f'HR@{k}'].append(hr)
    
    # Average metrics
    metrics = {}
    for k in k_list:
        metrics[f'NDCG@{k}'] = np.mean(all_ndcg[f'NDCG@{k}'])
        metrics[f'HR@{k}'] = np.mean(all_hr[f'HR@{k}'])
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix string for printing
    """
    print(f"\n{prefix}Evaluation Metrics:")
    print("-" * 60)
    for metric_name, value in sorted(metrics.items()):
        print(f"  {metric_name}: {value:.4f}")
    print("-" * 60)

