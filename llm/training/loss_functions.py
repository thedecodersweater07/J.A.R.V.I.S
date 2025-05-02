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
