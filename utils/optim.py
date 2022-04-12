import os

import torch
from torch import optim
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

class Optim(object):
    def __init__(self, **kwargs):
        params = kwargs
        self.lr = params.get('learning_rate', 1e-4)
        self.method = params.get('method', 'adamw')
        self.weight_decay = params.get('weight_decay', 0.0)
        self.lr_decay = params.get('lr_decay', 0.0)
        self.warmup_steps = params.get('warmup_steps', -1)

    def get_optimizer_scheduler(self, model, t_total):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        if self.method == 'sgd':
            optimizer = optim.SGD(optimizer_grouped_parameters, lr=self.lr)
        elif self.method == 'adam':
            optimizer = optim.Adam(optimizer_grouped_parameters, lr=self.lr)
        elif self.method == 'adamw':
            # optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        else:
            raise RuntimeError("invalid optim method: " + self.method)
        num_warmup_steps = self.warmup_steps if self.warmup_steps >= 0 else int(t_total * 0.2)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
        )
        return optimizer, scheduler
