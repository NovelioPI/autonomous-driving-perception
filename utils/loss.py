import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        self.criterion = nn.CrossEntropyLoss(reduction=reduction,
                                             weight=weight.float(),
                                             ignore_index=ignore_index)
            
    def forward(self, inputs, targets):
        b, c, h, w = inputs.size()
        targets = targets.view(-1)
        valid_mask = targets.ne(self.ignore_index)
        targets = targets * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(inputs, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)
        
        if self.min_kept <= num_valid:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[targets, torch.arange(len(targets), dtype=torch.long)]
            
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                else:
                    threshold = self.thresh
                kept_mask = mask_prob.le(threshold)
                targets = targets * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                
        targets = targets.masked_fill_(~valid_mask, self.ignore_index)
        targets = targets.view(b, h, w)
        return self.criterion(inputs, targets)