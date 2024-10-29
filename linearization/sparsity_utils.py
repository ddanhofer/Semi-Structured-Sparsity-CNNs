from enum import Enum
from typing import Tuple

class AugmentedChannel(Enum):
    NONE = 0
    INPUT = 1
    OUTPUT = 2
    
def augment(out_features, in_features) -> Tuple[AugmentedChannel, int, int, int]:
    if out_features * in_features % 4 == 0:
        return AugmentedChannel.NONE, out_features, in_features, out_features * in_features // 4
    
    # auxiliary functions for determining optimal padding
    _rnd_up = lambda x: (x // 4 + 1) * 4 
    _ops_ctr = lambda o, i: o * _rnd_up(i) < _rnd_up(o) * i
    
    channels = (out_features, in_features)
    if _ops_ctr(out_features, in_features):
        # padding the number of input channels causes less operations
        augmented_channel = AugmentedChannel.INPUT
        channels = (out_features, _rnd_up(in_features))    
    else:
        # padding the number of output channels causes less operations
        augmented_channel = AugmentedChannel.OUTPUT
        channels = (_rnd_up(out_features), in_features)
    num_quadruplets = channels[0] * channels[1] // 4
    
    return augmented_channel, channels[0], channels[1], num_quadruplets