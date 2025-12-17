import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from data_utils import get_loaders
from data_utils_stage2 import get_stage2_loaders
from models.cld_stage1 import CLDStage1
from models.cld_stage2 import CLDStage2
from evaluate import (
    evaluate_stage1, evaluate_stage2, evaluate_unbiased_stage1,
    create_unbiased_test_set, print_metrics
)


class Stage1Trainer:
    """Trainer for Stage 1: Causal Debiasing."""
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, lr=1e-4, alpha=0.1, beta=0.1, gamma=0.1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = defaultdict(float)
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Stage-1]')
        for batch in pbar:
            input_seq = batch['input_seq'].to(self.device)
            target = batch['target'].to(self.device)
            seq_len = batch['seq_len'].to(self.device)
            
            # Get text embeddings if available
            target_text_emb = None
            if 'target_text_emb' in batch:
                target_text_emb = batch['target_text_emb'].to(self.device)
            
            # Forward pass and compute loss
            self.optimizer.zero_grad()
            loss, loss_dict = self.model.compute_loss(
                input_seq, target, seq_len,
                target_text_emb=target_text_emb,
                alpha=self.alpha, beta=self.beta, gamma=self.gamma
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in loss_dict.items():
                loss_components[key] += value
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'L_rec': f'{loss_dict["L_rec"]:.4f}',
                'L_U': f'{loss_dict["L_U"]:.4f}',
                'L_I': f'{loss_dict["L_I"]:.4f}'
            })
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def evaluate(self):
        """Evaluate on validation set."""
        metrics = evaluate_stage1(self.model, self.val_loader, self.device)
        return metrics
    
    def save_checkpoint(self, epoch, save_path, metadata, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args': {
                'hidden_size': self.model.hidden_size,
                'num_blocks': len(self.model.matching_module.blocks),
                'num_heads': self.model.matching_module.blocks[0].attention.num_heads if len(self.model.matching_module.blocks) > 0 else 1,
                'dropout': 0.2,  # Default
                'max_len': self.model.max_len,
                'text_embedding_dim': 768 if self.model.use_semantic_alignment else None,
                'use_semantic_alignment': self.model.use_semantic_alignment
            },
            'metadata': metadata
        }
        torch.save(checkpoint, save_path)
        if is_best:
            best_path = save_path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)


class Stage2Trainer:
    """Trainer for Stage 2: LLM Integration."""
    
    def __init__(self, model, train_loader, val_loader, test_loader,
                 device, lr=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = torch.optim.Adam(model.projector.parameters(), lr=lr)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        self.model.projector.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Stage-2]')
        for batch in pbar:
            text_token_ids = torch.tensor(batch['text_token_ids'], dtype=torch.long, device=self.device)
            soft_token_positions = batch['soft_token_positions']
            soft_token_embeddings = batch['soft_token_embeddings']
            labels = batch['labels']
            attention_mask = batch['attention_mask']
            label_start_positions = batch['label_start_positions']
            
            # Convert soft_token_embeddings to tensors
            soft_token_embeddings_tensors = []
            for sample_embs in soft_token_embeddings:
                sample_tensors = [emb.to(self.device) if isinstance(emb, torch.Tensor) 
                                 else torch.tensor(emb, device=self.device) 
                                 for emb in sample_embs]
                soft_token_embeddings_tensors.append(sample_tensors)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
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
            torch.nn.utils.clip_grad_norm_(self.model.projector.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self):
        """Evaluate on validation set."""
        # For Stage-2, we use loss as the metric
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating Stage-2', leave=False):
                text_token_ids = torch.tensor(batch['text_token_ids'], dtype=torch.long, device=self.device)
                soft_token_positions = batch['soft_token_positions']
                soft_token_embeddings = batch['soft_token_embeddings']
                labels = batch['labels']
                attention_mask = batch['attention_mask']
                label_start_positions = batch['label_start_positions']
                
                soft_token_embeddings_tensors = []
                for sample_embs in soft_token_embeddings:
                    sample_tensors = [emb.to(self.device) if isinstance(emb, torch.Tensor) 
                                     else torch.tensor(emb, device=self.device) 
                                     for emb in sample_embs]
                    soft_token_embeddings_tensors.append(sample_tensors)
                
                outputs = self.model(
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
        return {'loss': avg_loss}
    
    def save_checkpoint(self, epoch, save_path, metadata, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'projector_state_dict': self.model.projector.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metadata': metadata
        }
        torch.save(checkpoint, save_path)
        if is_best:
            best_path = save_path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)


def train_stage1(args, save_dir):
    """Train Stage 1: Causal Debiasing."""
    print("\n" + "=" * 60)
    print("Stage 1: Causal Debiasing Training")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, metadata = get_loaders(
        data_path=args.data_path,
        dataset_name=args.dataset,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        max_len=args.max_len,
        batch_size=32,  # Stage-1 batch size
        num_workers=args.num_workers,
        extract_text_features=args.extract_text_features,
        sbert_model=args.sbert_model
    )
    
    print(f"Number of items: {metadata['num_items']}")
    print(f"Number of users: {metadata['num_users']}")
    
    # Initialize model
    print("\nInitializing Stage-1 model...")
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
    
    # Create trainer
    trainer = Stage1Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        lr=1e-4,  # Stage-1 learning rate
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )
    
    # Training loop
    best_val_ndcg = 0.0
    
    for epoch in range(1, args.stage1_epochs + 1):
        print(f"\nEpoch {epoch}/{args.stage1_epochs}")
        print("-" * 60)
        
        # Train
        avg_loss, loss_components = trainer.train_epoch(epoch)
        print(f"\nTraining Loss: {avg_loss:.4f}")
        print(f"  L_rec: {loss_components.get('L_rec', 0):.4f}")
        print(f"  L_U: {loss_components.get('L_U', 0):.4f}")
        print(f"  L_I: {loss_components.get('L_I', 0):.4f}")
        if 'L_matching' in loss_components:
            print(f"  L_matching: {loss_components['L_matching']:.4f}")
        
        # Evaluate
        if epoch % args.eval_every == 0:
            print("\nEvaluating on validation set...")
            val_metrics = trainer.evaluate()
            print_metrics(val_metrics, "Validation")
            
            # Save best model
            val_ndcg = val_metrics.get('NDCG@10', 0.0)
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                checkpoint_path = os.path.join(save_dir, f'stage1_{args.dataset}_best.pt')
                trainer.save_checkpoint(epoch, checkpoint_path, metadata, is_best=True)
                print(f"Saved best model to {checkpoint_path}")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(save_dir, f'stage1_{args.dataset}_epoch_{epoch}.pt')
            trainer.save_checkpoint(epoch, checkpoint_path, metadata)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    # Load best model
    best_checkpoint_path = os.path.join(save_dir, f'stage1_{args.dataset}_best.pt')
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    # Standard evaluation
    test_metrics = evaluate_stage1(model, test_loader, args.device)
    print_metrics(test_metrics, "Test (Standard)")
    
    # Unbiased evaluation
    print("\nCreating unbiased test set...")
    unbiased_test_set = create_unbiased_test_set(
        test_loader, 
        num_negatives=100,
        num_items=metadata['num_items']
    )
    
    unbiased_metrics = evaluate_unbiased_stage1(model, unbiased_test_set, args.device)
    print_metrics(unbiased_metrics, "Test (Unbiased)")
    
    return best_checkpoint_path


def train_stage2(args, stage1_checkpoint_path, save_dir):
    """Train Stage 2: LLM Integration."""
    print("\n" + "=" * 60)
    print("Stage 2: LLM Integration Training")
    print("=" * 60)
    
    # Load Stage-1 model
    print("\nLoading Stage-1 model...")
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
    
    # Load data
    print("\nLoading data for Stage-2...")
    train_loader, val_loader, test_loader, metadata = get_stage2_loaders(
        data_path=args.data_path,
        dataset_name=args.dataset,
        stage1_model_path=stage1_checkpoint_path,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        max_len=args.max_len,
        batch_size=4,  # Stage-2 batch size
        num_workers=args.num_workers,
        num_candidates=args.num_candidates,
        num_popular=args.num_popular,
        device=torch.device(args.device)
    )
    
    # Initialize Stage-2 model
    print("\nInitializing Stage-2 model...")
    model = CLDStage2(
        stage1_model=stage1_model,
        llm_model_name=args.llm_model_name,
        embedding_dim=args.embedding_dim
    )
    model = model.to(args.device)
    
    # Create trainer
    trainer = Stage2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        lr=1e-4  # Stage-2 learning rate
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.stage2_epochs + 1):
        print(f"\nEpoch {epoch}/{args.stage2_epochs}")
        print("-" * 60)
        
        # Train
        avg_loss = trainer.train_epoch(epoch)
        print(f"\nTraining Loss: {avg_loss:.4f}")
        
        # Evaluate
        if epoch % args.eval_every == 0:
            print("\nEvaluating on validation set...")
            val_metrics = trainer.evaluate()
            print(f"Validation Loss: {val_metrics['loss']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint_path = os.path.join(save_dir, f'stage2_{args.dataset}_best.pt')
                trainer.save_checkpoint(epoch, checkpoint_path, metadata, is_best=True)
                print(f"Saved best model to {checkpoint_path}")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(save_dir, f'stage2_{args.dataset}_epoch_{epoch}.pt')
            trainer.save_checkpoint(epoch, checkpoint_path, metadata)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    # Load best model
    best_checkpoint_path = os.path.join(save_dir, f'stage2_{args.dataset}_best.pt')
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=args.device)
        model.projector.load_state_dict(checkpoint['projector_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    test_metrics = trainer.evaluate()
    print(f"\nTest Loss: {test_metrics['loss']:.4f}")
    
    return best_checkpoint_path


