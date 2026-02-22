import math
from typing import Optional, Callable
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.data
                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(g, alpha=1 - beta1)        # m = β1*m + (1-β1)*g
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2) # v = β2*v + (1-β2)*g²

                # Bias-corrected learning rate
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # AdamW update with weight decay.
                p.data.mul_(1 - lr * wd)                          
                p.data += -alpha_t * (m / (v.sqrt() + eps))       

        return loss
