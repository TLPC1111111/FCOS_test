import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import logging
from pathlib import Path

from FCOS_model import FCOS, ResNet50Backbone, FPN, FCOSHead
from Loss import FCOSLoss, FocalLoss, GIoULoss, CenterNessLoss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FCOSTrainer:
    """Trainer class for FCOS object detection model."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model architecture components
        self.backbone = ResNet50Backbone(pretrained=args.pretrained)
        self.fpn = FPN(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
        self.head = FCOSHead(in_channels=256, num_classes=args.num_classes)
        
        # Build complete FCOS model
        self.model = FCOS(
            backbone=self.backbone,
            fpn=self.fpn,
            head=self.head,
            num_classes=args.num_classes
        )
        self.model = self.model.to(self.device)
        
        # Initialize loss functions
        self.focal_loss = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        self.giou_loss = GIoULoss()
        self.centerness_loss = CenterNessLoss()
        self.fcos_loss = FCOSLoss(
            focal_loss=self.focal_loss,
            giou_loss=self.giou_loss,
            centerness_loss=self.centerness_loss,
            num_classes=args.num_classes
        )
        
        # Initialize optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
        
        self.best_loss = float('inf')
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        logger.info(f"Starting epoch {epoch + 1}")
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            targets = [target.to(self.device) for target in targets]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss_dict = self.fcos_loss(outputs, targets)
            total_loss_value = sum(loss_dict.values())
            
            # Backward pass
            total_loss_value.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            total_loss += total_loss_value.item()
            num_batches += 1
            
            if (batch_idx + 1) % self.args.log_interval == 0:
                avg_loss = total_loss / num_batches
                logger.info(
                    f"Epoch [{epoch + 1}] Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {total_loss_value.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Classification Loss: {loss_dict['classification_loss']:.4f} | "
                    f"Bbox Loss: {loss_dict['bbox_loss']:.4f} | "
                    f"Centerness Loss: {loss_dict['centerness_loss']:.4f}"
                )
        
        avg_epoch_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        return avg_epoch_loss
    
    def validate(self, val_loader, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = [target.to(self.device) for target in targets]
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss_dict = self.fcos_loss(outputs, targets)
                total_loss_value = sum(loss_dict.values())
                
                total_loss += total_loss_value.item()
                num_batches += 1
        
        avg_val_loss = total_loss / num_batches
        logger.info(f"Validation Loss at Epoch {epoch + 1}: {avg_val_loss:.4f}")
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'args': self.args
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['loss']
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch'] + 1
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        start_epoch = 0
        
        # Load checkpoint if resuming
        if self.args.resume and (self.checkpoint_dir / 'checkpoint_latest.pth').exists():
            start_epoch = self.load_checkpoint(self.checkpoint_dir / 'checkpoint_latest.pth')
        
        for epoch in range(start_epoch, self.args.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader, epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best=is_best)
            
            # Save latest checkpoint for resuming
            latest_path = self.checkpoint_dir / 'checkpoint_latest.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': val_loss,
                'args': self.args
            }, latest_path)
            
            logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs} completed\n")


def main():
    parser = argparse.ArgumentParser(description='FCOS Object Detection Training')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=80,
                        help='Number of object classes (default: 80 for COCO)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained ResNet50 backbone')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--lr_step_size', type=int, default=30,
                        help='Learning rate scheduler step size')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Learning rate scheduler gamma')
    
    # Loss function arguments
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Alpha parameter for focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    
    # Checkpoint and logging arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (batches)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    logger.info("FCOS Training Script")
    logger.info(f"Arguments: {args}")
    
    # Initialize trainer
    trainer = FCOSTrainer(args)
    
    # TODO: Load your actual dataset here
    # Example structure:
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # For now, create dummy loaders for demonstration
    logger.warning("Using dummy data loaders - replace with actual dataset")
    train_loader = []
    val_loader = []
    
    # Start training
    if train_loader and val_loader:
        trainer.train(train_loader, val_loader)
    else:
        logger.error("Train and validation loaders are empty. Please provide actual datasets.")


if __name__ == '__main__':
    main()
