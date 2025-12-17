import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from data_utils import get_loaders
from models.cld_stage1 import CLDStage1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CLD-Rec Stage 1: Causal Debiasing Training')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='Games',
                       choices=['Games', 'Toys', 'Sports'],
                       help='Dataset name')
    parser.add_argument('--min_user_interactions', type=int, default=5,
                       help='Minimum interactions per user')
    parser.add_argument('--min_item_interactions', type=int, default=5,
                       help='Minimum interactions per item')
    
    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=50,
                       help='Hidden dimension size (default: 50)')
    parser.add_argument('--num_blocks', type=int, default=2,
                       help='Number of self-attention blocks')
    parser.add_argument('--num_heads', type=int, default=1,
                       help='Number of attention heads')
    parser.add_argument('--max_len', type=int, default=50,
                       help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Weight for user conformity loss L_U')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='Weight for item popularity loss L_I')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Weight for semantic alignment loss L_matching')
    parser.add_argument('--extract_text_features', action='store_true',
                       help='Extract text features using SBERT')
    parser.add_argument('--sbert_model', type=str, default='all-MiniLM-L6-v2',
                       help='SBERT model name for text feature extraction')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate every N epochs')
    
    return parser.parse_args()


def evaluate(model, data_loader, device, k=10):
    """
    Evaluate model using Hit Rate@K and NDCG@K metrics.
    
    Args:
        model: CLDStage1 model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        k: Top-K for evaluation
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    hits = []
    ndcgs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating', leave=False):
            input_seq = batch['input_seq'].to(device)
            target = batch['target'].to(device)
            seq_len = batch['seq_len'].to(device)
            
            # Get predictions
            logits = model.predict(input_seq, seq_len)  # (batch_size, num_items-1)
            
            # Get top-k predictions
            _, top_k_indices = torch.topk(logits, k, dim=1)  # (batch_size, k)
            top_k_indices = top_k_indices + 1  # Adjust for padding token
            
            # Compute Hit Rate@K and NDCG@K
            target_expanded = target.unsqueeze(1)  # (batch_size, 1)
            hits_batch = (top_k_indices == target_expanded).any(dim=1).float()
            hits.extend(hits_batch.cpu().numpy())
            
            # NDCG@K
            for i in range(len(target)):
                target_item = target[i].item()
                top_k_items = top_k_indices[i].cpu().numpy()
                
                if target_item in top_k_items:
                    rank = np.where(top_k_items == target_item)[0][0] + 1
                    ndcg = 1.0 / np.log2(rank + 1)
                else:
                    ndcg = 0.0
                ndcgs.append(ndcg)
    
    hit_rate = np.mean(hits)
    ndcg_score = np.mean(ndcgs)
    
    return {
        f'HR@{k}': hit_rate,
        f'NDCG@{k}': ndcg_score
    }


def train_epoch(model, train_loader, optimizer, device, alpha, beta, gamma):
    """
    Train for one epoch.
    
    Returns:
        Average loss and loss components
    """
    model.train()
    total_loss = 0.0
    loss_components = defaultdict(float)
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        input_seq = batch['input_seq'].to(device)
        target = batch['target'].to(device)
        seq_len = batch['seq_len'].to(device)
        
        # Get text embeddings if available
        target_text_emb = None
        if 'target_text_emb' in batch:
            target_text_emb = batch['target_text_emb'].to(device)
        
        # Forward pass and compute loss
        optimizer.zero_grad()
        loss, loss_dict = model.compute_loss(input_seq, target, seq_len, 
                                             target_text_emb=target_text_emb,
                                             alpha=alpha, beta=beta, gamma=gamma)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for key, value in loss_dict.items():
            loss_components[key] += value
        num_batches += 1
        
        # Update progress bar
        postfix_dict = {
            'loss': f'{loss.item():.4f}',
            'L_rec': f'{loss_dict["L_rec"]:.4f}',
            'L_U': f'{loss_dict["L_U"]:.4f}',
            'L_I': f'{loss_dict["L_I"]:.4f}'
        }
        if 'L_matching' in loss_dict:
            postfix_dict['L_match'] = f'{loss_dict["L_matching"]:.4f}'
        pbar.set_postfix(postfix_dict)
    
    # Average losses
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 60)
    print("CLD-Rec Stage 1: Causal Debiasing Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Alpha (L_U weight): {args.alpha}")
    print(f"Beta (L_I weight): {args.beta}")
    print(f"Gamma (L_matching weight): {args.gamma}")
    print(f"Extract text features: {args.extract_text_features}")
    if args.extract_text_features:
        print(f"SBERT model: {args.sbert_model}")
    print("=" * 60)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, metadata = get_loaders(
        data_path=args.data_path,
        dataset_name=args.dataset,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        extract_text_features=args.extract_text_features,
        sbert_model=args.sbert_model
    )
    
    print(f"Number of items: {metadata['num_items']}")
    print(f"Number of users: {metadata['num_users']}")
    
    # Initialize model
    print("\nInitializing model...")
    text_embedding_dim = metadata.get('text_embedding_dim', 768) if args.extract_text_features else 768
    model = CLDStage1(
        num_items=metadata['num_items'],
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_len=args.max_len,
        text_embedding_dim=text_embedding_dim,
        use_semantic_alignment=args.extract_text_features
    )
    model = model.to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\nStarting training...")
    best_val_ndcg = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        avg_loss, loss_components = train_epoch(
            model, train_loader, optimizer, args.device, args.alpha, args.beta, args.gamma
        )
        
        print(f"\nTraining Loss: {avg_loss:.4f}")
        print(f"  L_rec: {loss_components['L_rec']:.4f}")
        print(f"  L_U: {loss_components['L_U']:.4f}")
        print(f"  L_I: {loss_components['L_I']:.4f}")
        if 'L_matching' in loss_components:
            print(f"  L_matching: {loss_components['L_matching']:.4f}")
        
        # Evaluate
        if epoch % args.eval_every == 0:
            print("\nEvaluating on validation set...")
            val_metrics = evaluate(model, val_loader, args.device, k=10)
            print(f"Validation Metrics:")
            for metric, value in val_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Save best model
            if val_metrics['NDCG@10'] > best_val_ndcg:
                best_val_ndcg = val_metrics['NDCG@10']
                checkpoint_path = os.path.join(args.save_dir, f'best_stage1_{args.dataset}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'args': args,
                    'metadata': metadata
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.save_dir, f'stage1_{args.dataset}_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'metadata': metadata
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    # Load best model
    best_checkpoint_path = os.path.join(args.save_dir, f'best_stage1_{args.dataset}.pt')
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    test_metrics = evaluate(model, test_loader, args.device, k=10)
    print(f"\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()

