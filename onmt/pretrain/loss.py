import torch.nn as nn


def build_loss_compute(reduce=True):
    if reduce:
        criterion = nn.CrossEntropyLoss(reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')
    return FuseLossCompute(criterion)


class FuseLossCompute(nn.Module):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, criterion):
        super(FuseLossCompute, self).__init__()
        self.criterion = criterion

    def forward(self, output, target):

        loss = self.criterion(output, target)

        return loss


