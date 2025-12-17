import torch
import argparse
from data_utils import get_loaders
from models.cld_stage1 import CLDStage1


def test_model(args):
    """Test CLD Stage 1 model initialization and forward pass."""
    print("=" * 60)
    print("Testing CLD-Rec Stage 1 Model")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, metadata = get_loaders(
        data_path=args.data_path,
        dataset_name=args.dataset,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Number of items: {metadata['num_items']}")
    print(f"Number of users: {metadata['num_users']}")
    
    # Initialize model
    print("\nInitializing CLD Stage 1 model...")
    model = CLDStage1(
        num_items=metadata['num_items'],
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_len=args.max_len
    )
    model = model.to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    
    for batch in train_loader:
        input_seq = batch['input_seq'].to(args.device)
        target = batch['target'].to(args.device)
        seq_len = batch['seq_len'].to(args.device)
        
        print(f"\nBatch shape:")
        print(f"  Input sequence: {input_seq.shape}")
        print(f"  Target: {target.shape}")
        print(f"  Sequence length: {seq_len.shape}")
        
        # Test forward pass
        with torch.no_grad():
            # Test prediction
            logits = model.predict(input_seq, seq_len)
            print(f"\nPrediction logits shape: {logits.shape}")
            print(f"  Min: {logits.min().item():.4f}, Max: {logits.max().item():.4f}, Mean: {logits.mean().item():.4f}")
            
            # Test forward with components
            final_logits, y_k, y_u, y_i = model.forward(input_seq, seq_len, return_components=True)
            print(f"\nComponent outputs:")
            print(f"  Final logits shape: {final_logits.shape}")
            print(f"  Matching score (y_k) shape: {y_k.shape}")
            print(f"  User conformity (y_u) shape: {y_u.shape}")
            print(f"  Item popularity (y_i) shape: {y_i.shape}")
            print(f"  y_u range: [{y_u.min().item():.4f}, {y_u.max().item():.4f}]")
            print(f"  y_i range: [{y_i.min().item():.4f}, {y_i.max().item():.4f}]")
            
            # Test loss computation
            loss, loss_dict = model.compute_loss(input_seq, target, seq_len, alpha=0.1, beta=0.1)
            print(f"\nLoss components:")
            print(f"  L_rec: {loss_dict['L_rec']:.4f}")
            print(f"  L_U: {loss_dict['L_U']:.4f}")
            print(f"  L_I: {loss_dict['L_I']:.4f}")
            print(f"  Total loss: {loss_dict['total_loss']:.4f}")
        
        break  # Test with one batch only
    
    print("\n" + "=" * 60)
    print("Model test completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CLD-Rec Stage 1 Model')
    
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='Games',
                       choices=['Games', 'Toys', 'Sports'],
                       help='Dataset name')
    parser.add_argument('--min_user_interactions', type=int, default=5,
                       help='Minimum interactions per user')
    parser.add_argument('--min_item_interactions', type=int, default=5,
                       help='Minimum interactions per item')
    parser.add_argument('--hidden_size', type=int, default=50,
                       help='Hidden dimension size')
    parser.add_argument('--num_blocks', type=int, default=2,
                       help='Number of self-attention blocks')
    parser.add_argument('--num_heads', type=int, default=1,
                       help='Number of attention heads')
    parser.add_argument('--max_len', type=int, default=50,
                       help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    test_model(args)

