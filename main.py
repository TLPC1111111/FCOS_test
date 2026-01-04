import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VOC2012Dataset(Dataset):
    """VOC2012 Dataset loader for FCOS object detection."""
    
    def __init__(self, root_dir, year='2012', image_set='train', transforms_=None):
        """
        Args:
            root_dir: Path to VOC dataset root directory
            year: Dataset year (2012, 2007, etc.)
            image_set: 'train', 'val', or 'test'
            transforms_: Image transformations to apply
        """
        self.root_dir = Path(root_dir)
        self.year = year
        self.image_set = image_set
        self.transforms = transforms_
        self.voc_root = self.root_dir / f'VOC{year}'
        self.image_dir = self.voc_root / 'JPEGImages'
        self.annotation_dir = self.voc_root / 'Annotations'
        self.image_set_file = self.voc_root / 'ImageSets' / 'Main' / f'{image_set}.txt'
        
        # Load image IDs
        with open(self.image_set_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        # VOC classes
        self.classes = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                       'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = self.image_dir / f'{image_id}.jpg'
        annotation_path = self.annotation_dir / f'{image_id}.xml'
        
        # Load image
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        # Load annotations
        boxes = []
        labels = []
        
        import xml.etree.ElementTree as ET
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            label = obj.find('name').text
            labels.append(self.class_to_idx[label])
            
            bbox = obj.find('bndbox')
            x_min = float(bbox.find('xmin').text)
            y_min = float(bbox.find('ymin').text)
            x_max = float(bbox.find('xmax').text)
            y_max = float(bbox.find('ymax').text)
            
            boxes.append([x_min, y_min, x_max, y_max])
        
        if self.transforms:
            image = self.transforms(image)
        
        return {
            'image': image,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            'image_id': image_id
        }


class FCOSModel(nn.Module):
    """FCOS (Fully Convolutional One-Stage Object Detection) model with ResNet50 backbone."""
    
    def __init__(self, num_classes=21, backbone_pretrained=True):
        super(FCOSModel, self).__init__()
        self.num_classes = num_classes
        
        # ResNet50 backbone
        resnet50 = models.resnet50(pretrained=backbone_pretrained)
        self.backbone = nn.Sequential(*list(resnet50.children())[:-2])
        
        # Feature pyramid head
        self.fpn_c5_lateral = nn.Conv2d(2048, 256, kernel_size=1)
        self.fpn_c4_lateral = nn.Conv2d(1024, 256, kernel_size=1)
        self.fpn_c3_lateral = nn.Conv2d(512, 256, kernel_size=1)
        
        # Smoothing
        self.fpn_c5_output = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_c4_output = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_c3_output = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Head layers
        self.class_head = self._create_head(256, num_classes)
        self.reg_head = self._create_head(256, 4)
        self.center_head = self._create_head(256, 1)
        
    def _create_head(self, in_channels, out_channels, num_layers=4):
        """Create detection head with multiple convolutional layers."""
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, 256, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(256, out_channels, kernel_size=3, padding=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through FCOS model."""
        # Backbone
        c3 = self.backbone[0](x)  # layer1
        c4 = self.backbone[1](c3)  # layer2
        c5 = self.backbone[2](c4)  # layer3
        
        # FPN
        p5 = self.fpn_c5_lateral(c5)
        p4 = self.fpn_c4_lateral(c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.fpn_c3_lateral(c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        
        p5 = self.fpn_c5_output(p5)
        p4 = self.fpn_c4_output(p4)
        p3 = self.fpn_c3_output(p3)
        
        # Heads
        class_pred = self.class_head(p3)
        reg_pred = self.reg_head(p3)
        center_pred = self.center_head(p3)
        
        return {
            'class_pred': class_pred,
            'reg_pred': reg_pred,
            'center_pred': center_pred
        }


def collate_fn(batch):
    """Custom collate function for handling variable number of boxes."""
    images = torch.stack([item['image'] for item in batch])
    image_ids = [item['image_id'] for item in batch]
    
    boxes_list = [item['boxes'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    
    return {
        'images': images,
        'boxes': boxes_list,
        'labels': labels_list,
        'image_ids': image_ids
    }


class FCOSTrainer:
    """Trainer class for FCOS model."""
    
    def __init__(self, model, device, num_classes=21, checkpoint_dir='./checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.best_loss = float('inf')
        self.epoch = 0
        self.iterations = 0
        
    def compute_loss(self, predictions, targets):
        """Compute FCOS loss function."""
        class_pred = predictions['class_pred']
        reg_pred = predictions['reg_pred']
        center_pred = predictions['center_pred']
        
        # Placeholder loss computation
        class_loss = torch.tensor(0.0, device=self.device)
        reg_loss = torch.tensor(0.0, device=self.device)
        center_loss = torch.tensor(0.0, device=self.device)
        
        # For actual implementation, compute foreground/background classification
        # regression loss for bounding boxes, and centerness loss
        
        total_loss = class_loss + reg_loss + center_loss
        return total_loss
    
    def train_epoch(self, train_loader, optimizer, lr_scheduler):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(self.device)
            boxes = [b.to(self.device) for b in batch['boxes']]
            labels = [l.to(self.device) for l in batch['labels']]
            
            # Forward pass
            optimizer.zero_grad()
            predictions = self.model(images)
            
            # Compute loss
            loss = self.compute_loss(predictions, {'boxes': boxes, 'labels': labels})
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            self.iterations += 1
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f'Epoch [{self.epoch}] Batch [{batch_idx + 1}/{len(train_loader)}] '
                           f'Loss: {loss.item():.4f}')
        
        # Update learning rate
        if lr_scheduler:
            lr_scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(self.device)
                boxes = [b.to(self.device) for b in batch['boxes']]
                labels = [l.to(self.device) for l in batch['labels']]
                
                predictions = self.model(images)
                loss = self.compute_loss(predictions, {'boxes': boxes, 'labels': labels})
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def save_checkpoint(self, optimizer, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'iterations': self.iterations,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f'fcos_epoch_{self.epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f'Checkpoint saved: {checkpoint_path}')
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'fcos_best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f'Best checkpoint saved: {best_path}')
    
    def load_checkpoint(self, checkpoint_path, optimizer=None):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.iterations = checkpoint['iterations']
        logger.info(f'Checkpoint loaded: {checkpoint_path}')


def create_lr_scheduler(optimizer, num_epochs, num_batches_per_epoch):
    """Create learning rate scheduler with warmup and cosine annealing."""
    total_steps = num_epochs * num_batches_per_epoch
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def inference(model, image, device, confidence_threshold=0.5):
    """Run inference on a single image."""
    model.eval()
    
    with torch.no_grad():
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        image = image.unsqueeze(0).to(device)
        
        # Model forward pass
        predictions = model(image)
        
        class_pred = predictions['class_pred']
        reg_pred = predictions['reg_pred']
        center_pred = predictions['center_pred']
        
        # Post-processing (placeholder)
        detections = {
            'boxes': [],
            'scores': [],
            'labels': []
        }
    
    return detections


def visualize_detections(image, boxes, labels, scores=None, class_names=None, save_path=None):
    """Visualize detections on image."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    if class_names is None:
        class_names = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
                      'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                      'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                      'sheep', 'sofa', 'train', 'tvmonitor']
    
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        
        rect = patches.Rectangle((x_min, y_min), width, height,
                                linewidth=2, edgecolor=colors[label], facecolor='none')
        ax.add_patch(rect)
        
        label_text = class_names[label]
        if scores is not None:
            label_text += f': {scores[i]:.2f}'
        
        ax.text(x_min, y_min - 10, label_text, color=colors[label],
               fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_axis_off()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f'Visualization saved: {save_path}')
    
    plt.show()


def main():
    """Main training and evaluation pipeline."""
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Hyperparameters
    batch_size = 2  # Single GPU training with batch size 2
    num_epochs = 100
    initial_lr = 0.01
    num_classes = 21
    input_size = 640
    
    # Create model
    logger.info('Creating FCOS model with ResNet50 backbone...')
    model = FCOSModel(num_classes=num_classes, backbone_pretrained=True)
    
    # Create trainer
    trainer = FCOSTrainer(model, device, num_classes=num_classes)
    
    # Data transforms
    train_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    logger.info('Loading VOC2012 datasets...')
    try:
        train_dataset = VOC2012Dataset(root_dir='./data', year='2012', image_set='train',
                                      transforms_=train_transforms)
        val_dataset = VOC2012Dataset(root_dir='./data', year='2012', image_set='val',
                                    transforms_=val_transforms)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)
        
        logger.info(f'Training samples: {len(train_dataset)}')
        logger.info(f'Validation samples: {len(val_dataset)}')
    except Exception as e:
        logger.warning(f'Could not load VOC2012 dataset: {e}')
        logger.info('Skipping training - dataset not available')
        return
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    logger.info('Starting training...')
    training_history = {'epoch': [], 'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        trainer.epoch = epoch
        
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer, lr_scheduler)
        
        # Validate
        val_loss = trainer.validate(val_loader)
        
        # Log
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}] '
                   f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        training_history['epoch'].append(epoch)
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        
        # Save checkpoint
        is_best = val_loss < trainer.best_loss
        if is_best:
            trainer.best_loss = val_loss
        trainer.save_checkpoint(optimizer, is_best=is_best)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(training_history['epoch'], training_history['train_loss'], 'b-', label='Train Loss')
    plt.plot(training_history['epoch'], training_history['val_loss'], 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FCOS Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('./training_history.png')
    plt.close()
    logger.info('Training history saved: ./training_history.png')
    
    logger.info('Training completed!')


if __name__ == '__main__':
    main()
