import os
import argparse
import torch
from trainer import train_stage1, train_stage2
from evaluate import evaluate_stage1, evaluate_unbiased_stage1, create_unbiased_test_set, print_metrics
from data_utils import get_loaders
from models.cld_stage1 import CLDStage1
from models.sasrec import SASRec


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CLD-Rec: Unified Training and Evaluation Pipeline')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'train_stage1', 'train_stage2', 'test', 'test_data'],
                       help='Execution mode: train (both stages), train_stage1, train_stage2, test, or test_data')
    
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
                       help='Hidden dimension size')
    parser.add_argument('--num_blocks', type=int, default=2,
                       help='Number of self-attention blocks')
    parser.add_argument('--num_heads', type=int, default=1,
                       help='Number of attention heads')
    parser.add_argument('--max_len', type=int, default=50,
                       help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--llm_model_name', type=str, default='Qwen/Qwen1.5-1.8B',
                       help='LLM model name')
    parser.add_argument('--embedding_dim', type=int, default=50,
                       help='Dimension of Stage-1 embeddings')
    
    # Training arguments
    parser.add_argument('--stage1_epochs', type=int, default=50,
                       help='Number of Stage-1 training epochs')
    parser.add_argument('--stage2_epochs', type=int, default=10,
                       help='Number of Stage-2 training epochs')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Weight for user conformity loss')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='Weight for item popularity loss')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Weight for semantic alignment loss')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--num_candidates', type=int, default=10,
                       help='Number of candidate items for Stage-2')
    parser.add_argument('--num_popular', type=int, default=3,
                       help='Number of popular items for Stage-2')
    parser.add_argument('--extract_text_features', action='store_true',
                       help='Extract text features using SBERT')
    parser.add_argument('--sbert_model', type=str, default='all-MiniLM-L6-v2',
                       help='SBERT model name')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate every N epochs')
    parser.add_argument('--skip_stage1', action='store_true',
                       help='Skip Stage-1 training (use existing checkpoint)')
    parser.add_argument('--stage1_checkpoint', type=str, default=None,
                       help='Path to Stage-1 checkpoint (if skipping Stage-1 or for Stage-2)')
    parser.add_argument('--stage2_checkpoint', type=str, default=None,
                       help='Path to Stage-2 checkpoint (for testing)')
    
    return parser.parse_args()


def test_data_loading(args):
    """Test data loading and display statistics."""
    print("=" * 60)
    print("Testing Data Loading and Preprocessing")
    print("=" * 60)
    
    # Get data loaders
    train_loader, val_loader, test_loader, metadata = get_loaders(
        data_path=args.data_path,
        dataset_name=args.dataset,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        max_len=args.max_len,
        batch_size=32,
        num_workers=args.num_workers,
        extract_text_features=args.extract_text_features,
        sbert_model=args.sbert_model
    )
    
    print("\n" + "=" * 60)
    print("Data Statistics")
    print("=" * 60)
    print(f"Number of items (vocab size): {metadata['num_items']}")
    print(f"Number of users: {metadata['num_users']}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    print("\n" + "=" * 60)
    print("Testing Data Batch")
    print("=" * 60)
    for batch in train_loader:
        print(f"Input sequence shape: {batch['input_seq'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        print(f"Sequence length shape: {batch['seq_len'].shape}")
        print(f"Sample input sequence (first 10): {batch['input_seq'][0][-10:]}")
        print(f"Sample target: {batch['target'][0].item()}")
        print(f"Sample sequence length: {batch['seq_len'][0].item()}")
        if 'target_text_emb' in batch:
            print(f"Text embedding shape: {batch['target_text_emb'].shape}")
        break
    
    return train_loader, val_loader, test_loader, metadata


def test_model(args, metadata):
    """Test model initialization and forward pass."""
    print("\n" + "=" * 60)
    print("Testing Model Initialization")
    print("=" * 60)
    
    # Test SASRec
    print("\nTesting SASRec model...")
    sasrec_model = SASRec(
        num_items=metadata['num_items'],
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_len=args.max_len
    )
    sasrec_model = sasrec_model.to(args.device)
    print(f"SASRec parameters: {sum(p.numel() for p in sasrec_model.parameters()):,}")
    
    # Test CLDStage1
    print("\nTesting CLDStage1 model...")
    text_embedding_dim = metadata.get('text_embedding_dim', 768) if args.extract_text_features else 768
    stage1_model = CLDStage1(
        num_items=metadata['num_items'],
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_len=args.max_len,
        text_embedding_dim=text_embedding_dim,
        use_semantic_alignment=args.extract_text_features
    )
    stage1_model = stage1_model.to(args.device)
    print(f"CLDStage1 parameters: {sum(p.numel() for p in stage1_model.parameters()):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    dummy_input = torch.randint(1, metadata['num_items'], (batch_size, args.max_len)).to(args.device)
    dummy_seq_len = torch.randint(10, args.max_len, (batch_size,)).to(args.device)
    dummy_target = torch.randint(1, metadata['num_items'], (batch_size,)).to(args.device)
    
    with torch.no_grad():
        # Test SASRec
        sasrec_output = sasrec_model(dummy_input, dummy_seq_len)
        print(f"SASRec output shape: {sasrec_output.shape}")
        
        # Test CLDStage1
        stage1_logits = stage1_model.predict(dummy_input, dummy_seq_len)
        print(f"CLDStage1 prediction shape: {stage1_logits.shape}")
        
        loss, loss_dict = stage1_model.compute_loss(dummy_input, dummy_target, dummy_seq_len,
                                                     alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        print(f"CLDStage1 loss: {loss.item():.4f}")
        print(f"  L_rec: {loss_dict['L_rec']:.4f}")
        print(f"  L_U: {loss_dict['L_U']:.4f}")
        print(f"  L_I: {loss_dict['L_I']:.4f}")
    
    print("\nModel test completed successfully!")


def run_test(args):
    """Run evaluation on trained models."""
    print("=" * 60)
    print("CLD-Rec: Model Evaluation")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, metadata = get_loaders(
        data_path=args.data_path,
        dataset_name=args.dataset,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        max_len=args.max_len,
        batch_size=32,
        num_workers=args.num_workers,
        extract_text_features=args.extract_text_features,
        sbert_model=args.sbert_model
    )
    
    # Load Stage-1 model
    if args.stage1_checkpoint:
        stage1_checkpoint_path = args.stage1_checkpoint
    else:
        stage1_checkpoint_path = os.path.join(args.save_dir, f'stage1_{args.dataset}_best.pt')
    
    if not os.path.exists(stage1_checkpoint_path):
        print(f"Error: Stage-1 checkpoint not found at {stage1_checkpoint_path}")
        return
    
    print(f"\nLoading Stage-1 model from {stage1_checkpoint_path}...")
    checkpoint = torch.load(stage1_checkpoint_path, map_location=args.device)
    
    stage1_args = checkpoint.get('args', {})
    stage1_model = CLDStage1(
        num_items=checkpoint['metadata']['num_items'],
        hidden_size=stage1_args.get('hidden_size', 50),
        num_blocks=stage1_args.get('num_blocks', 2),
        num_heads=stage1_args.get('num_heads', 1),
        dropout=stage1_args.get('dropout', 0.2),
        max_len=stage1_args.get('max_len', 50),
        text_embedding_dim=stage1_args.get('text_embedding_dim', 768),
        use_semantic_alignment=stage1_args.get('use_semantic_alignment', False)
    )
    stage1_model.load_state_dict(checkpoint['model_state_dict'])
    stage1_model = stage1_model.to(args.device)
    stage1_model.eval()
    
    print("Stage-1 model loaded successfully")
    
    # Standard evaluation
    print("\n" + "=" * 60)
    print("Standard Evaluation")
    print("=" * 60)
    test_metrics = evaluate_stage1(stage1_model, test_loader, args.device)
    print_metrics(test_metrics, "Test (Standard)")
    
    # Unbiased evaluation
    print("\n" + "=" * 60)
    print("Unbiased Evaluation")
    print("=" * 60)
    print("Creating unbiased test set with uniform random sampling...")
    unbiased_test_set = create_unbiased_test_set(
        test_loader,
        num_negatives=100,
        num_items=metadata['num_items']
    )
    
    unbiased_metrics = evaluate_unbiased_stage1(stage1_model, unbiased_test_set, args.device)
    print_metrics(unbiased_metrics, "Test (Unbiased)")
    
    print("\nEvaluation completed!")


def main():
    """Main function."""
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("CLD-Rec: Unified Training and Evaluation Pipeline")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    if args.mode == 'test_data':
        # Test data loading and model initialization
        train_loader, val_loader, test_loader, metadata = test_data_loading(args)
        test_model(args, metadata)
        
    elif args.mode == 'train_stage1':
        # Train Stage-1 only
        print("\nTraining Stage-1 only...")
        train_stage1(args, args.save_dir)
        
    elif args.mode == 'train_stage2':
        # Train Stage-2 only
        if not args.stage1_checkpoint:
            args.stage1_checkpoint = os.path.join(args.save_dir, f'stage1_{args.dataset}_best.pt')
        if not os.path.exists(args.stage1_checkpoint):
            print(f"Error: Stage-1 checkpoint not found at {args.stage1_checkpoint}")
            return
        print(f"\nTraining Stage-2 with Stage-1 checkpoint: {args.stage1_checkpoint}")
        train_stage2(args, args.stage1_checkpoint, args.save_dir)
        
    elif args.mode == 'test':
        # Run evaluation
        run_test(args)
        
    elif args.mode == 'train':
        # Two-stage training
        print(f"Stage-1 epochs: {args.stage1_epochs}")
        print(f"Stage-2 epochs: {args.stage2_epochs}")
        
        # Stage 1 Training
        if args.skip_stage1:
            if args.stage1_checkpoint:
                stage1_checkpoint_path = args.stage1_checkpoint
            else:
                stage1_checkpoint_path = os.path.join(args.save_dir, f'stage1_{args.dataset}_best.pt')
            print(f"\nSkipping Stage-1 training. Using checkpoint: {stage1_checkpoint_path}")
        else:
            stage1_checkpoint_path = train_stage1(args, args.save_dir)
        
        # Stage 2 Training
        train_stage2(args, stage1_checkpoint_path, args.save_dir)
        
        print("\n" + "=" * 60)
        print("Training Pipeline Completed!")
        print("=" * 60)
    
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
