import sys
sys.path.insert(0, '..')


import torch
import torch.nn as nn

from typing import Tuple

from linearization.sparsity_utils import AugmentedChannel, augment


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

class Sparse2x4Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 1, groups: int = 1, bias: bool = False,
                 init: bool = False, gumbel_2x4: bool = False, augment_2x4: bool = False,
                 blocked: bool = False) -> None:
        super(Sparse2x4Conv2d, self).__init__()

        # ----- compute configuration -----
        self.blocked = blocked
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=1, padding=padding, stride=stride)
        self.use_bias = bias

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = in_channels * kernel_size * kernel_size
        self.out_features = out_channels

        # tracks the entropy of the layer and updates on forward pass
        self.entropy = None
        
        if in_channels % groups != 0:
            raise ValueError(f'Number of input channels ({in_channels}) must be divisible by the number of groups ({groups})!')
        if out_channels % groups != 0:
            raise ValueError(f'Number of output channels ({out_channels}) must be divisible by the number of groups ({groups})!')

        self.grouped_in_channels = in_channels // groups
        self.grouped_out_channels = out_channels // groups
        self.grouped_in_features = self.grouped_in_channels * kernel_size * kernel_size 
        self.grouped_out_features = self.grouped_out_channels

        # ----- 2-by-4 sparsity pattern via Gumbel-Softmax -----
        self.gumbel_2x4 = gumbel_2x4
        self.augment_2x4 = augment_2x4
        self.augmented_channel = AugmentedChannel.NONE
        
        
        self._grouped_out_features = self.grouped_out_features
        self._grouped_in_features = self.grouped_in_features
        
        if self.gumbel_2x4:    
            num_quadruplets = self.grouped_in_features * self.grouped_out_features // 4
            
            if not(self.augment_2x4) and num_quadruplets * 4 != self._grouped_in_features * self._grouped_out_features:
                raise ValueError(f'The number of in_features ({self._grouped_in_features}) * out_features ({self._grouped_out_features}) ' +\
                    f'= ({self.grouped_in_features * self.grouped_out_features}) must be divisible by 4!')
            elif self.augment_2x4:
                self.augmented_channel, self._grouped_out_features, self._grouped_in_features, num_quadruplets = augment(self.grouped_out_features, self.grouped_in_features)
            
            # there are 6 ways to choose 2 elements from a 4-tuple
            self.choice_weights = nn.Parameter(torch.ones(num_quadruplets * self.groups, 6), requires_grad=True) # initialize with all choices equally likely
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
        
        _W_block = lambda: nn.Parameter(torch.randn(self._grouped_out_features, self._grouped_in_features)) \
            if init else nn.Parameter(torch.zeros(self._grouped_out_features, self._grouped_in_features))
        if self.use_bias:
            _b_block = lambda: nn.Parameter(torch.randn(self._grouped_out_features)) \
                if init else nn.Parameter(torch.zeros(self._grouped_out_features))    
        if self.blocked:
            self.W = nn.Parameter(torch.block_diag(*[_W_block() for _ in range(self.groups)]))
            self.b = nn.Parameter(torch.cat([_b_block() for _ in range(self.groups)], dim=0)) if self.use_bias else None
        else:
            self.Ws = nn.ParameterList()
            self.bs = nn.ParameterList()
            self.zero_bias = nn.Parameter(torch.zeros(self._grouped_out_features, requires_grad=False))
            for _ in range(groups):    
                self.Ws.append(_W_block())
                if self.use_bias:
                    self.bs.append(_b_block())
        
       
    @property
    def weight(self):
        # return the weights as if they were from a Conv2d layer
        weights = []
        if self.blocked:
            for g in range(self.groups):
                offset_col_st = g * self._grouped_out_features
                offset_col_end = offset_col_st + self.grouped_out_features
                offset_row_st = g * self._grouped_in_features
                offset_row_end = offset_row_st + self.grouped_in_features
                
                _W = self.W[offset_col_st:offset_col_end, offset_row_st:offset_row_end]
                _W = _W.view((self.grouped_out_channels, self.grouped_in_channels, self.kernel_size, self.kernel_size))
                weights.append(_W)
        else:
            for W in self.Ws:
                _W = W[:self.grouped_out_features, :self.grouped_in_features]
                _W = _W.view((self.grouped_out_channels, self.grouped_in_channels, self.kernel_size, self.kernel_size))
                weights.append(_W)
        return torch.cat(weights, dim=0)

    @weight.setter
    def weight(self, value):
        # set the weights as if they were from a Conv2d layer // split along the channel dimension
        with torch.no_grad():
            blocks = []
            values = torch.chunk(value, self.groups, dim=0)
            for idx in range(len(values)):
                _values = values[idx].view((self.grouped_out_features, self.grouped_in_features))
                if self.augmented_channel == AugmentedChannel.INPUT:
                    _W = torch.zeros(self._grouped_out_features, self._grouped_in_features)
                    _W[:, :self.grouped_in_features] = _values
                elif self.augmented_channel == AugmentedChannel.OUTPUT:
                    _W = torch.zeros(self._grouped_out_features, self._grouped_in_features)
                    _W[:self.grouped_out_features, :] = _values
                else:
                    _W = _values
                blocks.append(_W)
            
        if self.blocked:
            self.W.data = torch.block_diag(*blocks)
        else:
            for W, _W in zip(self.Ws, blocks):
                W.data = _W

    @property
    def mean_entropy(self) -> float:
        if self.gumbel24:
            return sum([linear.mean_entropy for linear in self.linears]) / len(self.linears)
        return 0.0

    @property
    def bias(self):
        # return the bias as if it were from a Conv2d layer // stacked along the channel dimension
        if self.use_bias:
            biases = []
            if self.blocked:
                for g in range(self.groups):
                    offset_col_st = g * self._grouped_out_features
                    offset_col_end = offset_col_st + self.grouped_out_features
                    _b = self.b[offset_col_st:offset_col_end]
                    biases.append(_b)
            else:
                for b in self.bs:
                    _b = b[:self.grouped_out_features]
                    biases.append(_b)
            bias = torch.cat(biases, dim=0)
            return bias
        return None

    @bias.setter
    def bias(self, value):
        # set the bias as if it were from a Conv2d layer // split along the channel dimension
        with torch.no_grad():
            blocks = []
            values = torch.chunk(value, self.groups, dim=0)
            for idx in range(len(values)):
                _b = torch.zeros(self._grouped_out_features)
                _b[:self.grouped_out_features] = values[idx]
                blocks.append(_b)
            
            if self.blocked:
                self.b.data = torch.cat(blocks, dim=0)
            else:
                for b, _b in zip(self.bs, blocks):
                    b.data = _b
            
    @classmethod
    def from_torch_conv2d(cls, conv2d_layer, **kwargs):
        """
        Initialize a Sparse2x4Conv2d torch module (layer) on an existing nn.Conv2d module,
        copying its parameters and configuration.

        Args:
        conv2d_layer (nn.Conv2d): An instance of a pre-initialized nn.Conv2d layer.
        """

        if not isinstance(conv2d_layer, nn.Conv2d):
            raise TypeError("The provided layer is not an instance of nn.Conv2d: {}".format(type(conv2d_layer)))

        # copy configuration from the provided reference Conv2d layer
        in_channels = conv2d_layer.in_channels
        out_channels = conv2d_layer.out_channels
        if conv2d_layer.kernel_size[0] != conv2d_layer.kernel_size[1]:
            raise ValueError("Only square kernels are supported at the moment: {}".format(conv2d_layer.kernel_size))
        kernel_size = conv2d_layer.kernel_size[0]
        if conv2d_layer.stride[0] != conv2d_layer.stride[1]:
            raise ValueError("Only square strides are supported at the moment: {}".format(conv2d_layer.stride))
        stride = conv2d_layer.stride[0]
        if conv2d_layer.padding[0] != conv2d_layer.padding[1]:
            raise ValueError("Only square padding is supported at the moment: {}".format(conv2d_layer.padding))
        padding = conv2d_layer.padding[0]
        groups = conv2d_layer.groups

        bias = conv2d_layer.bias is not None
        
        layer = cls(in_channels, out_channels, kernel_size, stride, padding, groups, bias, False, **kwargs)
        
        with torch.no_grad():
            layer.weight = conv2d_layer.weight.data.clone()
            if bias:
                layer.bias = conv2d_layer.bias.data.clone()
            
        return layer

    def _forward_group(self, W: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_unfolded = self.unfold(x)
        if self.augmented_channel == AugmentedChannel.INPUT:
            W = W[:, :self.grouped_in_features]
        y = W.matmul(x_unfolded) + b.view(-1, 1)
        output = y[:, :self.grouped_out_features, :].view(*self.gr_output_shape)
        return output

    def forward(self, x: torch.Tensor):
        # ignore below assert to allow for compilation
        # assert len(x.shape) == 4, "Input tensor must be 4D (batch, channels, height, width)!"
        
        gr_output_height = (x.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        gr_output_width = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.gr_output_shape = (x.shape[0], self.grouped_out_channels, gr_output_height, gr_output_width)
        
        outputs = []
        x_chunks = torch.chunk(x, self.groups, dim=1)
        
        if self.gumbel_2x4:
            choice_mixture, entropy = _gumbel_softmax(self.choice_weights, hard=True, dim=-1) # shape (n_4blocks, 6)
            self.entropy = entropy
            mask = torch.matmul(choice_mixture, self.masking_patterns) # shape (num_blocks, 4)
            mask = mask.view(self.groups, self._grouped_out_features, self._grouped_in_features)
        
        if self.blocked:
            x_stacked = torch.cat([self.unfold(_x) for _x in x_chunks], dim=1)            
            W = self.W
            if self.gumbel_2x4:
                mask_diag = torch.block_diag(*[mask[i, :, :] for i in range(self.groups)])
                W = mask_diag * W
                if self.augmented_channel == AugmentedChannel.INPUT:
                    W = W[:, :self.in_features]

            if self.use_bias:
                output = W.matmul(x_stacked) + self.b.view(-1, 1)
            else:
                output = W.matmul(x_stacked)
            output = output[:, :self.out_channels, :].view(-1, self.out_channels, self.gr_output_shape[2], self.gr_output_shape[3])
        else:
            if self.use_bias:
                for idx, (W, b, _x) in enumerate(zip(self.Ws, self.bs, x_chunks)):
                    if self.gumbel_2x4:
                        W = W * mask[idx, :, :]
                    group_output = self._forward_group(W, b, _x)
                    outputs.append(group_output)
            else:
                for idx, (W, _x) in enumerate(zip(self.Ws, x_chunks)):
                    if self.gumbel_2x4:
                        W = W * mask[idx, :, :]
                    group_output = self._forward_group(W, self.zero_bias, _x)
                    outputs.append(group_output)

            output = torch.cat(outputs, dim=1)
            
        return output