import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import numpy as np


class ResNet50Backbone(nn.Module):
    """ResNet50 backbone network for FCOS"""
    
    def __init__(self, pretrained=True):
        super(ResNet50Backbone, self).__init__()
        self.resnet = resnet50(pretrained=pretrained)
        
        # Remove the classification layers
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
    
    def forward(self, x):
        """
        Extract multi-level features from ResNet50
        Returns features at different scales: C3, C4, C5
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c1 = self.layer1(x)   # stride=4, 256 channels
        c2 = self.layer2(c1)  # stride=8, 512 channels
        c3 = self.layer3(c2)  # stride=16, 1024 channels
        c4 = self.layer4(c3)  # stride=32, 2048 channels
        
        return c1, c2, c3, c4


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""
    
    def __init__(self, in_channels_list=[256, 512, 1024, 2048], out_channels=256):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        
        # Lateral layers to project backbone features to out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Smooth layers
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        
        # Additional layers for P6 and P7
        self.extra_convs = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features):
        """
        Args:
            features: list of backbone features [C1, C2, C3, C4]
        Returns:
            fpn_features: list of FPN features [P3, P4, P5, P6, P7]
        """
        # Project backbone features to out_channels
        laterals = [lateral(features[i]) for i, lateral in enumerate(self.lateral_convs)]
        
        # Build top-down path
        fpn_features = []
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')
        
        # Apply smooth convolutions
        fpn_features = [smooth(laterals[i]) for i, smooth in enumerate(self.smooth_convs)]
        
        # Add extra layers for P6 and P7
        p5 = fpn_features[-1]
        p6 = self.extra_convs[0:2](p5)
        p7 = self.extra_convs[2:4](p6)
        
        fpn_features.extend([p6, p7])
        
        return fpn_features


class FCOSHead(nn.Module):
    """FCOS detection head for classification, regression, and center-ness"""
    
    def __init__(self, in_channels=256, num_classes=80):
        super(FCOSHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Build classification branch
        self.cls_conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        
        # Build regression branch
        self.reg_conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.ctrness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: input feature map of shape (B, C, H, W)
        Returns:
            cls_logits: classification logits (B, num_classes, H, W)
            bbox_pred: bounding box predictions (B, 4, H, W)
            ctrness: center-ness predictions (B, 1, H, W)
        """
        cls_features = self.cls_conv_layers(x)
        cls_logits = self.cls_logits(cls_features)
        
        reg_features = self.reg_conv_layers(x)
        bbox_pred = torch.exp(self.bbox_pred(reg_features))
        ctrness = torch.sigmoid(self.ctrness(reg_features))
        
        return cls_logits, bbox_pred, ctrness


class FCOS(nn.Module):
    """Complete FCOS detector with ResNet50 backbone and FPN"""
    
    def __init__(self, num_classes=80, pretrained_backbone=True):
        super(FCOS, self).__init__()
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)
        
        # Feature Pyramid Network
        self.fpn = FPN(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
        
        # Detection heads for each FPN level
        self.head = FCOSHead(in_channels=256, num_classes=num_classes)
        
        # Strides for each FPN level
        self.strides = [8, 16, 32, 64, 128]
    
    def forward(self, x):
        """
        Forward pass of FCOS detector
        
        Args:
            x: input images (B, 3, H, W)
        
        Returns:
            cls_logits_list: list of classification logits for each FPN level
            bbox_pred_list: list of bbox predictions for each FPN level
            ctrness_list: list of center-ness predictions for each FPN level
        """
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Build FPN
        fpn_features = self.fpn(backbone_features)
        
        # Apply detection head to each FPN feature
        cls_logits_list = []
        bbox_pred_list = []
        ctrness_list = []
        
        for fpn_feature in fpn_features:
            cls_logits, bbox_pred, ctrness = self.head(fpn_feature)
            cls_logits_list.append(cls_logits)
            bbox_pred_list.append(bbox_pred)
            ctrness_list.append(ctrness)
        
        return cls_logits_list, bbox_pred_list, ctrness_list
    
    def inference(self, x, score_threshold=0.05, nms_threshold=0.6):
        """
        Inference mode with post-processing
        
        Args:
            x: input images (B, 3, H, W)
            score_threshold: confidence threshold for detection
            nms_threshold: NMS IoU threshold
        
        Returns:
            predictions: list of detected boxes and scores for each image
        """
        cls_logits_list, bbox_pred_list, ctrness_list = self.forward(x)
        
        batch_size = x.shape[0]
        predictions = [[] for _ in range(batch_size)]
        
        for fpn_level, (cls_logits, bbox_pred, ctrness) in enumerate(
            zip(cls_logits_list, bbox_pred_list, ctrness_list)
        ):
            stride = self.strides[fpn_level]
            predictions = self._decode_predictions(
                cls_logits, bbox_pred, ctrness, stride, 
                predictions, score_threshold
            )
        
        # Apply NMS
        final_predictions = [
            self._nms(preds, nms_threshold) if len(preds) > 0 else []
            for preds in predictions
        ]
        
        return final_predictions
    
    def _decode_predictions(self, cls_logits, bbox_pred, ctrness, stride, 
                           predictions, score_threshold):
        """Decode predictions from network outputs"""
        batch_size, num_classes, height, width = cls_logits.shape
        
        # Get class scores and class indices
        cls_probs = F.softmax(cls_logits, dim=1)
        scores, class_ids = cls_probs.max(dim=1)
        
        # Combine with center-ness
        scores = scores * ctrness.squeeze(1)
        
        # Process each image in batch
        for b in range(batch_size):
            # Get locations above threshold
            mask = scores[b] > score_threshold
            locations = torch.where(mask)
            
            if locations[0].numel() == 0:
                continue
            
            # Get coordinates
            y_coords = locations[0].float()
            x_coords = locations[1].float()
            
            # Convert to image coordinates
            cx = (x_coords + 0.5) * stride
            cy = (y_coords + 0.5) * stride
            
            # Get predictions at these locations
            l = bbox_pred[b, 0, y_coords, x_coords]
            r = bbox_pred[b, 1, y_coords, x_coords]
            t = bbox_pred[b, 2, y_coords, x_coords]
            b_val = bbox_pred[b, 3, y_coords, x_coords]
            
            # Convert to box format [x1, y1, x2, y2]
            x1 = cx - l
            y1 = cy - t
            x2 = cx + r
            y2 = cy + b_val
            
            # Collect predictions
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            scores_filtered = scores[b][mask]
            class_ids_filtered = class_ids[b][mask]
            
            for box, score, class_id in zip(boxes, scores_filtered, class_ids_filtered):
                predictions[b].append({
                    'box': box.cpu().detach().numpy(),
                    'score': score.cpu().detach().item(),
                    'class_id': class_id.cpu().detach().item()
                })
        
        return predictions
    
    def _nms(self, predictions, nms_threshold=0.6):
        """
        Non-Maximum Suppression (NMS) post-processing
        
        Args:
            predictions: list of predictions with 'box', 'score', 'class_id'
            nms_threshold: IoU threshold for NMS
        
        Returns:
            filtered_predictions: list of predictions after NMS
        """
        if len(predictions) == 0:
            return []
        
        # Sort by score
        predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
        
        keep = []
        while len(predictions) > 0:
            current = predictions.pop(0)
            keep.append(current)
            
            if len(predictions) == 0:
                break
            
            # Calculate IoU with remaining predictions
            current_box = current['box']
            remaining_boxes = np.array([p['box'] for p in predictions])
            
            ious = self._compute_iou(current_box, remaining_boxes)
            
            # Keep only predictions with IoU below threshold
            predictions = [
                p for p, iou in zip(predictions, ious) 
                if iou < nms_threshold
            ]
        
        return keep
    
    @staticmethod
    def _compute_iou(box, boxes):
        """
        Compute IoU between one box and multiple boxes
        
        Args:
            box: single box [x1, y1, x2, y2]
            boxes: multiple boxes (N, 4)
        
        Returns:
            ious: IoU values for each box
        """
        # Compute intersection area
        x1_inter = np.maximum(box[0], boxes[:, 0])
        y1_inter = np.maximum(box[1], boxes[:, 1])
        x2_inter = np.minimum(box[2], boxes[:, 2])
        y2_inter = np.minimum(box[3], boxes[:, 3])
        
        inter_w = np.maximum(0, x2_inter - x1_inter)
        inter_h = np.maximum(0, y2_inter - y1_inter)
        inter_area = inter_w * inter_h
        
        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area
        
        # Compute IoU
        ious = inter_area / (union_area + 1e-6)
        
        return ious


def nms(boxes, scores, threshold=0.6):
    """
    Standalone NMS post-processing function
    
    Args:
        boxes: tensor of shape (N, 4) in format [x1, y1, x2, y2]
        scores: tensor of shape (N,) with confidence scores
        threshold: IoU threshold for NMS
    
    Returns:
        keep_indices: indices of boxes to keep
    """
    if boxes.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long)
    
    # Calculate areas
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by score
    order = scores.argsort(descending=True)
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Calculate IoU
        inter_x1 = torch.max(x1[i], x1[order[1:]])
        inter_y1 = torch.max(y1[i], y1[order[1:]])
        inter_x2 = torch.min(x2[i], x2[order[1:]])
        inter_y2 = torch.min(y2[i], y2[order[1:]])
        
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        union_area = areas[i] + areas[order[1:]] - inter_area
        ious = inter_area / (union_area + 1e-6)
        
        # Filter by IoU threshold
        mask = ious < threshold
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long)


if __name__ == "__main__":
    # Example usage
    model = FCOS(num_classes=80, pretrained_backbone=True)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 512, 512)
    
    # Forward pass
    with torch.no_grad():
        cls_logits, bbox_pred, ctrness = model(dummy_input)
        print(f"Classification logits levels: {len(cls_logits)}")
        print(f"BBox predictions levels: {len(bbox_pred)}")
        print(f"Center-ness levels: {len(ctrness)}")
        
        # Inference with NMS
        predictions = model.inference(dummy_input)
        print(f"Predictions: {predictions}")
