import torch
import torch.nn as nn

class SupervisedContrastiveLoss(nn.Module):
    """
    Implementation of the Supervised Contrastive Loss (SCL) from Khosla et al. 2020.
    """
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        Compute the supervised contrastive loss.
        
        Args:
            features (torch.Tensor): Batch of feature vectors (latent representations)
                                    Shape: [batch_size, feature_dim]
            labels (torch.Tensor): Batch of labels. Shape: [batch_size]
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        features = nn.functional.normalize(features, p=2, dim=1)
        
        similarity_matrix = torch.matmul(features, features.T)
        
        similarity_matrix = similarity_matrix / self.temperature
        
        labels_expanded = labels.expand(batch_size, batch_size)
        mask_positive = labels_expanded.eq(labels_expanded.T)
        mask_self = torch.eye(batch_size, dtype=torch.bool, device=device)
        
        mask_positive = mask_positive & (~mask_self)
        
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        exp_logits = torch.exp(similarity_matrix - logits_max.detach())
        
        mask_valid_denominator = ~mask_self
        
        log_prob = torch.zeros(batch_size, device=device)
        
        valid_anchors = mask_positive.sum(1) > 0
        
        if valid_anchors.sum() > 0:
            exp_logits_positive = exp_logits * mask_positive
            numerator = exp_logits_positive.sum(1)
            
            denominator = exp_logits * mask_valid_denominator
            denominator = denominator.sum(1)
            
            log_prob[valid_anchors] = torch.log(numerator[valid_anchors] / denominator[valid_anchors])
            loss = -log_prob.sum() / valid_anchors.sum()
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss
