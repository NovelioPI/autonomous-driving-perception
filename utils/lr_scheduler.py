import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupPolyLR(_LRScheduler):
    def __init__(
        self, optimizer, T_max,
        warmup_factor=1.0 / 3, warmup_iters=500,
        eta_min=0, power=0.9, last_epoch=-1
    ):
        if T_max <= 0:
            raise ValueError("T_max must be > 0")
        if warmup_iters < 0:
            raise ValueError("warmup_iters must be >= 0")
        if warmup_iters > T_max:
            raise ValueError("warmup_iters must be <= T_max")
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.power = power
        self.T_max, self.eta_min = T_max, eta_min
        self.cur_iter = 0
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        # Clamp cur_iter to [0, T_max]
        cur_iter = min(max(self.cur_iter, 0), self.T_max)
        
        if cur_iter <= self.warmup_iters and self.warmup_iters > 0:
            alpha = cur_iter / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            poly_iter = min(max(cur_iter, self.warmup_iters), self.T_max)
            if self.T_max == self.warmup_iters:
                poly_factor = 0
            else:
                poly_factor = (1 - (poly_iter - self.warmup_iters) / (self.T_max - self.warmup_iters))
            poly_factor = max(poly_factor, 0)
            return [
                self.eta_min + (base_lr - self.eta_min) * math.pow(poly_factor, self.power)
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            self.cur_iter += 1
        else:
            self.cur_iter = epoch
        super().step(epoch)
        # PyTorch schedulers do not return lr in step()

