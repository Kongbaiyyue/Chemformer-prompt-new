import torch
import torch.nn as nn
import pytorch_lightning as pl


class PromptModel(pl.LightningModule):
    def __init__(
        self,
        prompt_model,
        bart_model
    ):
        super().__init__()

        self.prompt_model = prompt_model
        self.bart_model = bart_model

    def forward(self, x):
        raise NotImplementedError()