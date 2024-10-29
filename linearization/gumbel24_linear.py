# Author: P. Belcák
# source: https://gist.github.com/pbelcak/5100508aae912d4f809e090aa2bcf315
# copied from the source and modified to fit the project
# copied on 03/19/2024

import torch
from torch import nn
from typing import Tuple

class Gumbel24Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True, init: bool = False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=True)
        else:
            self.bias = None

        n_4blocks = in_features*out_features // 4
        if n_4blocks*4 != in_features*out_features:
            raise ValueError(f'The number in_features ({in_features}) * out_features ({out_features}) = ({in_features*out_features}) must be divisible by 4!')
        
        # there are 6 ways to choose 2 elements from a 4-tuple
        self.choice_weights = nn.Parameter(torch.ones(n_4blocks, 6), requires_grad=True) # initialize with all choices equally likely
        self.masking_patterns = nn.Parameter(\
            torch.tensor([
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0]
            ]),
            requires_grad=False
        )

        if init:
            nn.init.normal_(self.bias, std=1e-6)
            nn.init.xavier_uniform_(self.weight)
        else:
            pass # this assumes that the weights will be loaded into the layer from some pre-trained source
        
        # tracks the entropy of the layer and updates on forward pass
        self.entropy = None
    
    @classmethod
    def from_torch_linear(cls, linear: nn.Linear)  -> 'Gumbel24Linear':
        if not isinstance(linear, nn.Linear):
            raise TypeError("The provided layer is not an instance of nn.Linear: {}".format(type(linear)))
        
        bias = linear.bias is not None
        gumbel24 = cls(linear.in_features, linear.out_features, bias=bias, init=False)
        
        gumbel24.weight.data = linear.weight.data
        if bias:
            gumbel24.bias.data = linear.bias.data

        return gumbel24

    def forward(self, x):
        masking_pattern_choice_mixture, entropy = _gumbel_softmax(self.choice_weights, hard=True, dim=-1) # shape (n_4blocks, 6)
        self.entropy = entropy
        mask = torch.matmul(masking_pattern_choice_mixture, self.masking_patterns) # shape (n_4blocks, 4)

        # mask the weight matrix
        mask = mask.reshape(self.out_features, self.in_features)
        masked_weight = self.weight * mask

        # compute the output
        return torch.nn.functional.linear(x, masked_weight, self.bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
    
    @property
    def mean_entropy(self) -> float:
        if self.entropy is None:
            return 0.0
        return self.entropy.mean().item()


def _gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    gumbels = (
        -torch.empty_like(logits).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.softmax(dim)

    # estimating the entropy of the gumbel softmax distribution (accurate in the limit of tau -> 0)
    entropy = -(y_soft * y_soft.log()).sum(dim)
    
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft # straight through estimation trick
    else:
        ret = y_soft
    
    return ret, entropy