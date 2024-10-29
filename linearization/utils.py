import torch

def linearize_dict(d: dict) -> dict:
    conv_adjustments = {}
    for i in range(1, 4):
        key = f'conv{i}.weight'
        value = f'conv{i}.linear.weight'
        conv_adjustments[key] = value

    downsampling_adjustments = {}
    for i in range(0, 2):
        key = f'downsample.{i}.weight'
        value = f'downsample.{i}.linear.weight'
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

def linearize_dict_for_vgg(d: dict) -> dict:
    feature_adjustments = {}
    for i in range(0, 110):
        key = f'features.{i}.weight'
        value = f'features.{i}.linear.weight'
        feature_adjustments[key] = value
        
        key = f'features.{i}.bias'
        value = f'features.{i}.linear.bias'
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
        
def _is_group(key, next_key):
    if 'linears.' in key and 'linears.' in next_key:
        # find characters at the end of linears. until next '.'
        key_number = int(key.split('linears.')[1].split('.')[0])
        next_key_number = int(next_key.split('linears.')[1].split('.')[0])
        
        if int(key_number) + 1 == int(next_key_number):
            return True
    return False

def _find_parameter_groups(src_network, target_network) -> list:
    lin_keys = [pair[0] for pair in target_network.named_parameters()]
    orig_keys = [pair[0] for pair in src_network.named_parameters()]

    lin_ptr = 0
    orig_ptr = 0
    
    groups = []
    group_ptr = None
    tracking_group = False

    while lin_ptr < len(lin_keys) and orig_ptr < len(orig_keys):
        lin_key = lin_keys[lin_ptr]
        orig_key = orig_keys[orig_ptr]
        
        if 'running' in orig_key:
            print(f'running')

        if _is_group(lin_key, lin_keys[min([lin_ptr + 1, len(lin_keys) - 1])]):
            if not tracking_group:
                tracking_group = True
                groups.append(Group(orig_keys[orig_ptr]))
                group_ptr = len(groups) - 1
            groups[group_ptr].add_entry(lin_keys[lin_ptr])
            lin_ptr += 1
        else:
            if tracking_group:
                tracking_group = False
                groups[group_ptr].add_entry(lin_keys[lin_ptr])
            else:
                pair_group = Group(orig_keys[orig_ptr])
                pair_group.add_entry(lin_keys[lin_ptr])
                groups.append(pair_group)
            lin_ptr += 1
            orig_ptr += 1

    return groups

def _state_dict_from_groups(g, state) -> dict:
    d = {}
    for group in g:
        length = group.length
        
        src = group.src
        weights = state[src]
        
        if length == 1:
            d[group.target[0]] = weights.view(weights.shape[0], -1)
            if d[group.target[0]].shape[1] == 1:
                d[group.target[0]] = d[group.target[0]].squeeze()
        else:
            chunks = torch.chunk(weights, length, dim=0)
            for i, target in enumerate(group.target):
                d[target] = chunks[i].view(chunks[i].shape[0], -1)
                if d[target].shape[1] == 1:
                    d[target] = d[target].squeeze()
    for key in state.keys():
        if 'running_mean' in key or 'running_var' in key:
            d[key] = state[key]
    return d

def linearize_dict_for_shufflenet(original_state: dict, original_arch, linearized_arch) -> dict:
    groups = _find_parameter_groups(original_arch, linearized_arch)
    state_dict = _state_dict_from_groups(groups, original_state)
    return state_dict

def _effnet_state_dict_from_groups(g, state) -> dict:
    d = {}
    for group in g:
        length = group.length
        
        src = group.src
        weights = state[src]
        
        if length == 1:
            if 'fc' in group.target[0]:
                d[group.target[0]] = weights
            else:
                d[group.target[0]] = weights.view(weights.shape[0], -1)
                if d[group.target[0]].shape[1] == 1:
                    d[group.target[0]] = d[group.target[0]].squeeze()
        else:
            chunks = torch.chunk(weights, length, dim=0)
            for i, target in enumerate(group.target):
                if 'fc' in target:
                    d[target] = chunks[i]
                else:
                    d[target] = chunks[i].view(chunks[i].shape[0], -1)
                    if d[target].shape[1] == 1:
                        d[target] = d[target].squeeze()
    for key in state.keys():
        if 'running_mean' in key or 'running_var' in key:
            d[key] = state[key]
    return d

def linearize_dict_for_effnet(original_state: dict, original_arch, linearized_arch) -> dict:
    groups = _find_parameter_groups(original_arch, linearized_arch)
    state_dict = _effnet_state_dict_from_groups(groups, original_state)
    return state_dict