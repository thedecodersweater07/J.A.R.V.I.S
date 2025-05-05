import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothedCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        smooth_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        return torch.mean(torch.sum(-smooth_hot * F.log_softmax(pred, dim=1), dim=1))

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings_a, embeddings_b):
        normalized_a = F.normalize(embeddings_a, p=2, dim=1)
        normalized_b = F.normalize(embeddings_b, p=2, dim=1)
        similarity_matrix = torch.matmul(normalized_a, normalized_b.t()) / self.temperature
        return F.cross_entropy(similarity_matrix, torch.arange(len(embeddings_a)).to(embeddings_a.device))

class AdaptiveLoss(nn.Module):
    """Loss function with adaptive temperature"""
    def __init__(self, initial_temperature=0.07, adaptation_rate=0.01):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        self.adaptation_rate = adaptation_rate
        
    def forward(self, predictions, targets):
        self._adapt_temperature(predictions)
        scaled_preds = predictions / self.temperature
        return F.cross_entropy(scaled_preds, targets)
        
    def _adapt_temperature(self, predictions):
        with torch.no_grad():
            confidence = F.softmax(predictions, dim=1).max(1)[0].mean()
            self.temperature.data *= (1 + self.adaptation_rate * (confidence - 0.5))
