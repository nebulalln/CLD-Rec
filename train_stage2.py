import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data_utils_stage2 import get_stage2_loaders
from models.cld_stage2 import CLDStage2
from models.cld_stage1 import CLDStage1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CLD-Rec Stage 2: LLM Integration Training')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='Games',
                       choices=['Games', 'Toys', 'Sports'],
                       help='Dataset name')
    parser.add_argument('--stage1_model_path', type=str, required=True,
                       help='Path to trained Stage-1 model checkpoint')
    parser.add_argument('--min_user_interactions', type=int, default=5,
                       help='Minimum interactions per user')
    parser.add_argument('--min_item_interactions', type=int, default=5,
                       help='Minimum interactions per item')
    
    # Model arguments
    parser.add_argument('--llm_model_name', type=str, default='Qwen/Qwen1.5-1.8B',
                       help='LLM model name')
    parser.add_argument('--embedding_dim', type=int, default=50,
                       help='Dimension of Stage-1 embeddings')
    parser.add_argument('--max_len', type=int, default=50,
                       help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (small for LLM)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loader workers')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate for projector')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--num_candidates', type=int, default=10,
                       help='Number of candidate items')
    parser.add_argument('--num_popular', type=int, default=3,
                       help='Number of popular items')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=2,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate every N epochs')
    
    return parser.parse_args()


def train_epoch(model, train_loader, optimizer, device):
    """
    Train for one epoch.
    
    Returns:
        Average loss
    """
    model.train()
    # Only enable gradients for projector
    model.projector.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        text_token_ids = torch.tensor(batch['text_token_ids'], dtype=torch.long, device=device)
        soft_token_positions = batch['soft_token_positions']
        soft_token_embeddings = batch['soft_token_embeddings']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        label_start_positions = batch['label_start_positions']
        
        # Convert soft_token_embeddings to tensors
        soft_token_embeddings_tensors = []
        for sample_embs in soft_token_embeddings:
            sample_tensors = [emb.to(device) if isinstance(emb, torch.Tensor) else torch.tensor(emb, device=device) 
                             for emb in sample_embs]
            soft_token_embeddings_tensors.append(sample_tensors)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            text_token_ids=text_token_ids,
            soft_token_positions=soft_token_positions,
            soft_token_embeddings=soft_token_embeddings_tensors,
            attention_mask=attention_mask,
            labels=labels,
            label_start_positions=label_start_positions
        )
        
        loss = outputs.loss
        
        # Backward pass (only projector has gradients)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.projector.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, data_loader, device):
    """
    Evaluate model.
    
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating', leave=False):
            text_token_ids = torch.tensor(batch['text_token_ids'], dtype=torch.long, device=device)
            soft_token_positions = batch['soft_token_positions']
            soft_token_embeddings = batch['soft_token_embeddings']
            labels = batch['labels']
            attention_mask = batch['attention_mask']
            label_start_positions = batch['label_start_positions']
            
            # Convert soft_token_embeddings to tensors
            soft_token_embeddings_tensors = []
            for sample_embs in soft_token_embeddings:
                sample_tensors = [emb.to(device) if isinstance(emb, torch.Tensor) else torch.tensor(emb, device=device) 
                                 for emb in sample_embs]
                soft_token_embeddings_tensors.append(sample_tensors)
            
            outputs = model(
                text_token_ids=text_token_ids,
                soft_token_positions=soft_token_positions,
                soft_token_embeddings=soft_token_embeddings_tensors,
                attention_mask=attention_mask,
                labels=labels,
                label_start_positions=label_start_positions
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 60)
    print("CLD-Rec Stage 2: LLM Integration Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"LLM Model: {args.llm_model_name}")
    print(f"Stage-1 Model: {args.stage1_model_path}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load Stage-1 model
    print("\nLoading Stage-1 model...")
    checkpoint = torch.load(args.stage1_model_path, map_location=args.device)
    
    # Reconstruct Stage-1 model
    stage1_args = checkpoint.get('args', {})
    if isinstance(stage1_args, argparse.Namespace):
        stage1_model = CLDStage1(
            num_items=checkpoint['metadata']['num_items'],
            hidden_size=stage1_args.hidden_size,
            num_blocks=stage1_args.num_blocks,
            num_heads=stage1_args.num_heads,
            dropout=stage1_args.dropout,
            max_len=stage1_args.max_len,
            text_embedding_dim=stage1_args.get('text_embedding_dim', 768),
            use_semantic_alignment=stage1_args.get('use_semantic_alignment', False)
        )
    else:
        # Fallback: use defaults
        stage1_model = CLDStage1(
            num_items=checkpoint['metadata']['num_items'],
            hidden_size=50,
            num_blocks=2,
            num_heads=1,
            dropout=0.2,
            max_len=50
        )
    
    stage1_model.load_state_dict(checkpoint['model_state_dict'])
    stage1_model = stage1_model.to(args.device)
    stage1_model.eval()
    
    print("Stage-1 model loaded successfully")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, metadata = get_stage2_loaders(
        data_path=args.data_path,
        dataset_name=args.dataset,
        stage1_model_path=args.stage1_model_path,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_candidates=args.num_candidates,
        num_popular=args.num_popular,
        device=torch.device(args.device)
    )
    
    print(f"Number of items: {metadata['num_items']}")
    print(f"Number of users: {metadata['num_users']}")
    
    # Initialize Stage-2 model
    print("\nInitializing Stage-2 model...")
    model = CLDStage2(
        stage1_model=stage1_model,
        llm_model_name=args.llm_model_name,
        embedding_dim=args.embedding_dim
    )
    model = model.to(args.device)
    
    # Verify only projector is trainable
    trainable_params = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters (Projector): {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Initialize optimizer (only for projector)
    optimizer = torch.optim.Adam(model.projector.parameters(), lr=args.lr)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, args.device)
        print(f"\nTraining Loss: {avg_loss:.4f}")
        
        # Evaluate
        if epoch % args.eval_every == 0:
            print("\nEvaluating on validation set...")
            val_loss = evaluate(model, val_loader, args.device)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(args.save_dir, f'best_stage2_{args.dataset}.pt')
                torch.save({
                    'epoch': epoch,
                    'projector_state_dict': model.projector.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'args': args,
                    'metadata': metadata
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.save_dir, f'stage2_{args.dataset}_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'projector_state_dict': model.projector.state_dict(),
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
    best_checkpoint_path = os.path.join(args.save_dir, f'best_stage2_{args.dataset}.pt')
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path)
        model.projector.load_state_dict(checkpoint['projector_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    test_loss = evaluate(model, test_loader, args.device)
    print(f"\nTest Loss: {test_loss:.4f}")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()

