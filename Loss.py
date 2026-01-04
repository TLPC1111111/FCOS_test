import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FocalLoss(nn.Module):
    """
    Focal Loss for classification.
    Addresses class imbalance by down-weighting easy examples and focusing on hard negatives.
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Args:
            alpha (float): Weighting factor in [0, 1] to balance positive/negative examples.
            gamma (float): Exponent of the modulating factor (1 - p_t)^gamma.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions (Tensor): Shape (N, C) - raw logits from classification head.
            targets (Tensor): Shape (N,) - ground truth class indices.
        
        Returns:
            Tensor: Scalar loss value.
        """
        # Get probabilities
        p = F.softmax(predictions, dim=1)
        
        # Get class probabilities
        class_mask = torch.zeros_like(predictions)
        class_mask.scatter_(1, targets.view(-1, 1), 1)
        
        # Get the probabilities of the true class
        probs = (p * class_mask).sum(dim=1)
        
        # Compute focal loss
        log_p = F.log_softmax(predictions, dim=1)
        log_probs = (log_p * class_mask).sum(dim=1)
        
        # Focal loss = -alpha * (1 - p_t)^gamma * log(p_t)
        focal_weight = (1 - probs) ** self.gamma
        focal_loss = -self.alpha * focal_weight * log_probs
        
        return focal_loss.mean()


class GIoULoss(nn.Module):
    """
    GIoU Loss for bounding box regression.
    Improves upon IoU loss by considering the smallest enclosing box.
    
    Reference: https://arxiv.org/abs/1902.09630
    """
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): 'mean' or 'sum' - how to reduce the loss.
        """
        super(GIoULoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes (Tensor): Shape (N, 4) - predicted boxes in (x1, y1, x2, y2) format.
            target_boxes (Tensor): Shape (N, 4) - target boxes in (x1, y1, x2, y2) format.
        
        Returns:
            Tensor: Scalar loss value.
        """
        # Calculate areas
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        
        # Calculate intersection
        lt = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
        rb = torch.min(pred_boxes[:, 2:], target_boxes[:, 2:])
        wh = (rb - lt).clamp(min=0)
        intersection = wh[:, 0] * wh[:, 1]
        
        # Calculate union
        union = pred_area + target_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        # Calculate enclosing box
        enclose_lt = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
        enclose_rb = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
        enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
        
        # Calculate GIoU
        giou = iou - (enclose_area - union) / (enclose_area + 1e-6)
        
        # Calculate loss
        loss = 1.0 - giou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CenterNessLoss(nn.Module):
    """
    Center-ness Loss for detection quality improvement.
    Improves detection quality by focusing on objects near the center of their receptive field.
    
    Reference: https://arxiv.org/abs/2103.14030
    """
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): 'mean' or 'sum' - how to reduce the loss.
        """
        super(CenterNessLoss, self).__init__()
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, centerness_pred, centerness_targets):
        """
        Args:
            centerness_pred (Tensor): Shape (N,) - predicted centerness logits.
            centerness_targets (Tensor): Shape (N,) - ground truth centerness targets [0, 1].
        
        Returns:
            Tensor: Scalar loss value.
        """
        loss = self.bce_loss(centerness_pred, centerness_targets)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FCOSLoss(nn.Module):
    """
    Combined Loss for FCOS (Fully Convolutional One-Stage Object Detection).
    Combines Focal Loss, GIoU Loss, and Center-ness Loss.
    """
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, 
                 lambda_cls=1.0, lambda_reg=1.0, lambda_center=1.0):
        """
        Args:
            focal_alpha (float): Alpha parameter for Focal Loss.
            focal_gamma (float): Gamma parameter for Focal Loss.
            lambda_cls (float): Weight for classification loss.
            lambda_reg (float): Weight for regression loss.
            lambda_center (float): Weight for center-ness loss.
        """
        super(FCOSLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.giou_loss = GIoULoss(reduction='mean')
        self.centerness_loss = CenterNessLoss(reduction='mean')
        
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.lambda_center = lambda_center
    
    def forward(self, cls_pred, reg_pred, center_pred, 
                cls_targets, reg_targets, center_targets, 
                valid_mask=None):
        """
        Calculate combined FCOS loss.
        
        Args:
            cls_pred (Tensor): Shape (N, C) - classification predictions (logits).
            reg_pred (Tensor): Shape (N, 4) - regression predictions (ltrb or x1y1x2y2).
            center_pred (Tensor): Shape (N,) - center-ness predictions (logits).
            cls_targets (Tensor): Shape (N,) - classification targets (class indices).
            reg_targets (Tensor): Shape (N, 4) - regression targets.
            center_targets (Tensor): Shape (N,) - center-ness targets [0, 1].
            valid_mask (Tensor, optional): Shape (N,) - mask for valid samples.
        
        Returns:
            dict: Dictionary containing individual loss components and total loss.
        """
        if valid_mask is not None:
            cls_pred = cls_pred[valid_mask]
            reg_pred = reg_pred[valid_mask]
            center_pred = center_pred[valid_mask]
            cls_targets = cls_targets[valid_mask]
            reg_targets = reg_targets[valid_mask]
            center_targets = center_targets[valid_mask]
        
        # Classification loss (Focal Loss)
        loss_cls = self.focal_loss(cls_pred, cls_targets) * self.lambda_cls
        
        # Regression loss (GIoU Loss)
        loss_reg = self.giou_loss(reg_pred, reg_targets) * self.lambda_reg
        
        # Center-ness loss
        loss_center = self.centerness_loss(center_pred, center_targets) * self.lambda_center
        
        # Total loss
        loss_total = loss_cls + loss_reg + loss_center
        
        return {
            'loss_total': loss_total,
            'loss_cls': loss_cls.detach(),
            'loss_reg': loss_reg.detach(),
            'loss_center': loss_center.detach()
        }


def calculate_centerness(ltrb_targets):
    """
    Calculate center-ness targets from ltrb (left, top, right, bottom) format.
    
    Args:
        ltrb_targets (Tensor): Shape (N, 4) - targets in ltrb format.
    
    Returns:
        Tensor: Shape (N,) - center-ness targets in [0, 1].
    """
    left = ltrb_targets[:, 0]
    top = ltrb_targets[:, 1]
    right = ltrb_targets[:, 2]
    bottom = ltrb_targets[:, 3]
    
    # Center-ness = sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))
    centerness = torch.sqrt(
        (torch.min(left, right) / (torch.max(left, right) + 1e-6)) *
        (torch.min(top, bottom) / (torch.max(top, bottom) + 1e-6))
    )
    
    return centerness


# Example usage
if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize loss module
    loss_fn = FCOSLoss(focal_alpha=0.25, focal_gamma=2.0,
                       lambda_cls=1.0, lambda_reg=1.0, lambda_center=1.0)
    loss_fn = loss_fn.to(device)
    
    # Create dummy data
    batch_size = 16
    num_classes = 80
    
    cls_pred = torch.randn(batch_size, num_classes, device=device)
    reg_pred = torch.randn(batch_size, 4, device=device).abs()
    center_pred = torch.randn(batch_size, device=device)
    
    cls_targets = torch.randint(0, num_classes, (batch_size,), device=device)
    reg_targets = torch.randn(batch_size, 4, device=device).abs()
    center_targets = torch.rand(batch_size, device=device)
    
    # Calculate loss
    losses = loss_fn(cls_pred, reg_pred, center_pred,
                     cls_targets, reg_targets, center_targets)
    
    print("Loss Components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
