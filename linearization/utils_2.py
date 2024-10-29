from typing import Optional
import torch
from torch import nn
import numpy as np

from linearization.sparsity_utils import augment
from linearization.sparse_2x4_conv2d import Sparse2x4Conv2d

def rename_weights_resnet(d: dict, blocked: bool = False) -> dict:
    conv_adjustments = {}
    for i in range(1, 4):
        key = f'conv{i}.weight'
        value = f'conv{i}.Ws.0' if not(blocked) else f'conv{i}.W'
        conv_adjustments[key] = value

    downsampling_adjustments = {}
    for i in range(0, 2):
        key = f'downsample.{i}.weight'
        value = f'downsample.{i}.Ws.0' if not(blocked) else f'downsample.{i}.W'
        downsampling_adjustments[key] = value

    key_adjustments = {
        **conv_adjustments,
        **downsampling_adjustments
    }
    
    new_d = {}
    for key, value in d.items():
        new_key = str(key)
        for target, replacement in key_adjustments.items():
            new_key = new_key.replace(target, replacement)
        if key != new_key:
            try:
                new_shape = [value.shape[0], value.shape[1] * value.shape[2] * value.shape[3]]
                new_d[new_key] = value.view(new_shape)
                new_d[key] = value
            except Exception as e:
                new_d[new_key] = value
                new_d[key] = value
        else:
            new_d[key] = value
        # print(f'{key} -> {new_key}')
    
    return new_d

def rename_weights_vgg(d: dict, blocked: bool = False) -> dict:
    feature_adjustments = {}
    for i in range(0, 110):
        key = f'features.{i}.weight'
        value = f'features.{i}.Ws.0' if not(blocked) else f'features.{i}.W'
        feature_adjustments[key] = value
        
        key = f'features.{i}.bias'
        value = f'features.{i}.bs.0' if not(blocked) else f'features.{i}.b'
        feature_adjustments[key] = value

    key_adjustments = {
        **feature_adjustments,
    }
    
    new_d = {}
    for key, value in d.items():
        # print(key)
        new_key = str(key)
        for target, replacement in key_adjustments.items():
            new_key = new_key.replace(target, replacement)
        if key != new_key:
            try:
                new_shape = [value.shape[0], value.shape[1] * value.shape[2] * value.shape[3]]
                new_d[new_key] = value.view(new_shape)
                new_d[key] = value
            except Exception as e:
                new_d[new_key] = value
                new_d[key] = value
        else:
            new_d[key] = value
        # print(f'{key} -> {new_key}')
    
    return new_d

class Group:
    def __init__(self, src):
        self.src = src
        self.length = 0
        self.target = []
    
    def add_entry(self, target):
        self.length += 1
        self.target.append(target)
        
def _is_group(key, next_key, blocked: Optional[bool] = False):
    if 'Ws.' in key and 'Ws.' in next_key:
        # find characters at the end of linears. until next '.'
        key_number = int(key.split('Ws.')[1].split('.')[0])
        next_key_number = int(next_key.split('Ws.')[1].split('.')[0])
        
        if int(key_number) + 1 == int(next_key_number):
            return True
    return False

def _find_parameter_groups(src_network, target_network, blocked: Optional[bool] = False) -> list:
    sparse_keys = [pair[0] for pair in target_network.named_parameters()]
    orig_keys = [pair[0] for pair in src_network.named_parameters()]

    sparse_ptr = 0
    orig_ptr = 0
    
    groups = []
    group_ptr = None
    tracking_group = False

    while sparse_ptr < len(sparse_keys) and orig_ptr < len(orig_keys):
        sparse_key = sparse_keys[sparse_ptr]
        orig_key = orig_keys[orig_ptr]
        
        _skip = False
        for skip in ['zero_bias', 'choice_weights', 'masking_patterns']:
            if skip in sparse_key:
                _skip = True
                break
        if _skip:
            sparse_ptr += 1
            continue

        if _is_group(sparse_key, sparse_keys[min([sparse_ptr + 1, len(sparse_keys) - 1])]):
            if not tracking_group:
                tracking_group = True
                groups.append(Group(orig_keys[orig_ptr]))
                group_ptr = len(groups) - 1
            groups[group_ptr].add_entry(sparse_keys[sparse_ptr])
            sparse_ptr += 1
        else:
            if tracking_group:
                tracking_group = False
                groups[group_ptr].add_entry(sparse_keys[sparse_ptr])
            else:
                pair_group = Group(orig_keys[orig_ptr])
                pair_group.add_entry(sparse_keys[sparse_ptr])
                groups.append(pair_group)
            sparse_ptr += 1
            orig_ptr += 1

    return groups

def _efficientnet_state_dict_from_groups(g, state, sparse_arch, augment_2x4: Optional[bool] = False, blocked: Optional[bool] = False) -> dict:
    d = {}
    sparse_d = sparse_arch.state_dict()
    
    for group in g:
        length = group.length
        
        src = group.src
        weights = state[src]
        
        if length == 1 and blocked:
            if 'fc' in group.target[0]:
                d[group.target[0]] = weights
            else:
                if blocked:
                    if np.prod(weights.shape) == np.prod(sparse_d[group.target[0]].shape):
                        d[group.target[0]] = weights.view(weights.shape[0], -1)
                    else:
                        # grouped conv -> effnet only contains grouped convs
                        # where the number of groups is the number of output channels
                        num_groups = weights.shape[0]
                        diag_entries = torch.chunk(weights, num_groups, dim=0)
                        resh_entries = []
                        for i, entry in enumerate(diag_entries):
                            if augment_2x4:
                                out_channels = entry.shape[0]
                                in_channels = np.prod(entry.shape[1:])
                                _, _out_channels, _in_channels, _ = augment(out_channels, in_channels)
                                
                                _zero_tensor = torch.zeros(_out_channels, _in_channels)
                                _zero_tensor[:out_channels, :in_channels] = entry.view(out_channels, -1)
                                resh_entries.append(_zero_tensor)
                            else:
                                resh_entries.append(entry.view(entry.shape[0], -1))
                        
                        reshaped = torch.block_diag(*resh_entries)
                        d[group.target[0]] = reshaped
                        
                    if d[group.target[0]].shape[1] == 1:
                        d[group.target[0]] = d[group.target[0]].squeeze()
        else:
            chunks = torch.chunk(weights, length, dim=0)
            for i, target in enumerate(group.target):
                if 'fc' in target:
                    d[target] = chunks[i]
                else:
                    d[target] = chunks[i].view(chunks[i].shape[0], -1)
                    if augment_2x4:
                        out_channels = d[target].shape[0]
                        in_channels = d[target].shape[1]
                        _, _out_channels, _in_channels, _ = augment(out_channels, in_channels)
                        _zero_tensor = torch.zeros(_out_channels, _in_channels)
                        _zero_tensor[:out_channels, :in_channels] = d[target]
                        d[target] = _zero_tensor
                    if d[target].shape[1] == 1:
                        d[target] = d[target].squeeze()
    
    for key in state.keys():
        if 'running_mean' in key or 'running_var' in key:
            d[key] = state[key]
    return d

def rename_weights_efficientnet(original_state: dict, original_arch, sparse_arch, 
                                augment_2x4: Optional[bool] = False, blocked: Optional[bool] = False) -> dict:
    groups = _find_parameter_groups(original_arch, sparse_arch, blocked=blocked)
    state_dict = _efficientnet_state_dict_from_groups(groups, original_state, sparse_arch, augment_2x4=augment_2x4, blocked=blocked)
    return state_dict

def _shufflenet_state_dict_from_groups(g, state, sparse_arch, augment_2x4: Optional[bool] = False, blocked: Optional[bool] = False) -> dict:
    d = {}
    sparse_d = sparse_arch.state_dict()
    
    for group in g:
        length = group.length
        
        src = group.src
        weights = state[src]
        
        if weights.dim() == 1 and weights.shape == sparse_d[group.target[0]].shape:
            d[group.target[0]] = weights
            continue
            
        if length == 1 and blocked:
            if 'fc' in group.target[0]:
                d[group.target[0]] = weights
            else:
                if blocked:
                    if np.prod(weights.shape) == np.prod(sparse_d[group.target[0]].shape):
                        d[group.target[0]] = weights.view(weights.shape[0], -1)
                    else:            
                        # grouped conv -> effnet only contains grouped convs
                        # where the number of groups is the number of output channels
                        num_groups = weights.shape[0]
                        diag_entries = torch.chunk(weights, num_groups, dim=0)
                        resh_entries = []
                        for i, entry in enumerate(diag_entries):
                            if augment_2x4:
                                out_channels = entry.shape[0]
                                in_channels = np.prod(entry.shape[1:])
                                _, _out_channels, _in_channels, _ = augment(out_channels, in_channels)
                                
                                _zero_tensor = torch.zeros(_out_channels, _in_channels)
                                _zero_tensor[:out_channels, :in_channels] = entry.view(out_channels, -1)
                                resh_entries.append(_zero_tensor)
                            else:
                                resh_entries.append(entry.view(entry.shape[0], -1))
                        
                        reshaped = torch.block_diag(*resh_entries)
                        d[group.target[0]] = reshaped
                        
                    if d[group.target[0]].shape[1] == 1:
                        d[group.target[0]] = d[group.target[0]].squeeze()
        else:
            chunks = torch.chunk(weights, length, dim=0)
            for i, target in enumerate(group.target):
                if 'fc' in target:
                    d[target] = chunks[i]
                else:
                    d[target] = chunks[i].view(chunks[i].shape[0], -1)
                    if augment_2x4:
                        out_channels = d[target].shape[0]
                        in_channels = d[target].shape[1]
                        _, _out_channels, _in_channels, _ = augment(out_channels, in_channels)
                        _zero_tensor = torch.zeros(_out_channels, _in_channels)
                        _zero_tensor[:out_channels, :in_channels] = d[target]
                        d[target] = _zero_tensor
                    if d[target].shape[1] == 1:
                        d[target] = d[target].squeeze()
    
    for key in state.keys():
        if 'running_mean' in key or 'running_var' in key:
            d[key] = state[key]
    
    return d

def rename_weights_shufflenet(original_state: dict, original_arch, sparse_arch, 
                                augment_2x4: Optional[bool] = False, blocked: Optional[bool] = False) -> dict:
    groups = _find_parameter_groups(original_arch, sparse_arch, blocked=blocked)
    state_dict = _shufflenet_state_dict_from_groups(groups, original_state, sparse_arch, augment_2x4=augment_2x4, blocked=blocked)
    return state_dict

def compute_sparse_state_dict_next(original, sparse, debug=False, augment_2x4=False, ignore_grouped=False) -> dict:
    d2 = dict()
    k1s = iter(original.state_dict().keys())
    k2s = iter(sparse.state_dict().keys())
    
    while True:
        k1 = next(k1s, None)
        k2 = next(k2s, None)
        
        if k1 is None:
            break
        
        while 'choice_weights' in k2 or 'masking_patterns' in k2:
            k2 = next(k2s)
    
        orig_shape = original.state_dict()[k1].shape
        sparse_shape = sparse.state_dict()[k2].shape
        
        if orig_shape == sparse_shape:
            d2[k2] = original.state_dict()[k1]
            
        # this is a grouped convolution of 7x7 in ConvNeXt
        elif orig_shape[2] == 7 and orig_shape[3] == 7 and not(ignore_grouped):
            _dim = orig_shape[0]
            with torch.no_grad():
                _conv = nn.Conv2d(_dim, _dim, kernel_size=7,
                          padding=3, groups=_dim, bias=True)
                _conv.weight.copy_(original.state_dict()[k1])
                _sparse = Sparse2x4Conv2d.from_torch_conv2d(_conv, blocked=True, augment_2x4=augment_2x4, gumbel_2x4=augment_2x4)

                reshaped = _sparse.W
                d2[k2] = reshaped
        else:
            if len(orig_shape) == 4 and len(sparse_shape) == 2:
                try:
                    reshaped = original.state_dict()[k1].reshape((sparse_shape[0],sparse_shape[1],))
                    d2[k2] = reshaped
                except Exception as e:
                    if debug:
                        print(f'E: {e}')
                    else:
                        raise e
            else:
                if debug:
                    print(f'{k1} {original.state_dict()[k1].shape} -- {k2} {sparse.state_dict()[k2].shape}')
                else:
                    raise Exception(f'{k1} {original.state_dict()[k1].shape} -- {k2} {sparse.state_dict()[k2].shape}')
    return d2