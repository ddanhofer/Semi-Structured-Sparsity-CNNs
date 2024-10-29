# script based on pytorch/vision/references/classification/train.py
#   https://github.com/pytorch/vision/blob/5181a854d8b127cf465cd22a67c1b5aaf6ccae05/references/classification/train.py
#   accessed on 6th April 2024
#   source code is subject to the below license (https://github.com/pytorch/vision/blob/5181a854d8b127cf465cd22a67c1b5aaf6ccae05/LICENSE)
# the original source has been modified
'''
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import sys
sys.path.insert(0, './')

import datetime
import os
import time
import warnings

import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from transforms import get_mixup_cutmix
import presets

from resnet.resnet import resnet18, resnet34, resnet50
from resnet.resnet_sparse import resnet18 as resnet18_sparse
from resnet.resnet_sparse import resnet34 as resnet34_sparse
from resnet.resnet_sparse import resnet50 as resnet50_sparse
from linearization.utils_2 import rename_weights_resnet

from vgg.vgg import vgg11, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from vgg.vgg_sparse import vgg11 as vgg11_sparse
from vgg.vgg_sparse import vgg11_bn as vgg11_bn_sparse
from vgg.vgg_sparse import vgg13_bn as vgg13_bn_sparse
from vgg.vgg_sparse import vgg16_bn as vgg16_bn_sparse
from vgg.vgg_sparse import vgg19_bn as vgg19_bn_sparse
from linearization.utils_2 import rename_weights_vgg

from efficientnet.efficientnet import efficientnet_v2_s
from efficientnet.efficientnet_sparse import efficientnet_v2_s as efficientnet_v2_s_sparse
from linearization.utils_2 import rename_weights_efficientnet

from convnext.convnext import convnext_tiny, convnext_small
from convnext.convnext_sparse import convnext_tiny as convnext_tiny_sparse
from convnext.convnext_sparse import convnext_small as convnext_small_sparse
from linearization.utils_2 import compute_sparse_state_dict_next

from data_set.imagenet_1k import Imagenet1k
from data_set.imagenet_dali import get_data_loaders_dali

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None, global_metric_logger=None):
    model.train()
    
    if not(global_metric_logger is None):
        metric_logger = global_metric_logger
    else:
        metric_logger = utils.MetricLogger(delimiter="  ", wandb=args.use_wandb)
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, data_pair in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        if args.enable_dali:
            image, target = data_pair[0]["data"], data_pair[0]["label"].squeeze().long()
        else:
            image, target = data_pair
        
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        
        if args.channels_last:
            image = image.to(memory_format=torch.channels_last)
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)
        
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        # iterate through the model layers and compute the mean entropy of the gumbel softmax layers
        # if args.trainable == 'gumbel_2x4':
        #     entropies = []
        #     for _, module in model.named_modules():
        #         if isinstance(module, Gumbel24Linear):
        #             entropies.append(module.mean_entropy)
        #     mean_entropy = sum(entropies) / len(entropies)
        #     metric_logger.update(entropy=mean_entropy)
        
        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(tr_loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc1=acc1.item(), acc5=acc5.item(), imgs=(batch_size / (time.time() - start_time)))
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix="", global_metric_logger=None):
    model.eval()
    if not(global_metric_logger is None):
        metric_logger = global_metric_logger
    else:
        metric_logger = utils.MetricLogger(delimiter="  ", wandb=args.use_wandb)
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    context_mgr = torch.no_grad() if ('vgg' in args.model) else torch.inference_mode()
    with context_mgr:
        for data_pair in metric_logger.log_every(data_loader, print_freq, header):
            if args.enable_dali:
                image, target = data_pair[0]["data"], data_pair[0]["label"].squeeze().long()
            else:
                image, target = data_pair
                
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(val_loss=loss.item())
            metric_logger.update(val_acc1=acc1.item())
            metric_logger.update(val_acc5=acc5.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if not args.enable_dali:
        if (
            hasattr(data_loader.dataset, "__len__")
            and len(data_loader.dataset) != num_processed_samples
            and torch.distributed.get_rank() == 0
        ):
            # See FIXME above
            warnings.warn(
                f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
                "samples were used for the validation, which might bias the results. "
                "Try adjusting the batch size and / or the world size. "
                "Setting the world size to 1 is always a safe bet."
            )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(args):
    # Data loading code
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)
    
    print("Loading data from cache presupposing huggingface-cli login!")
    print("Using own Imagenet1k utility. Ignoring the provided arguments regarding directories, caching etc.")

    print("Loading training data!")
    st = time.time()
    
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    ra_magnitude = getattr(args, "ra_magnitude", None)
    augmix_severity = getattr(args, "augmix_severity", None)
    train_transform = presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
            ra_magnitude=ra_magnitude,
            augmix_severity=augmix_severity,
            backend=args.backend,
            use_v2=args.use_v2,
        )
    dataset = Imagenet1k(split='train', transform=train_transform)
    print("Took", time.time() - st)

    print("Loading validation data")
    preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
                use_v2=args.use_v2,
            )
    dataset_test = Imagenet1k(split='validation', transform=preprocessing)
    print("Took", time.time() - st)
    
    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)
    
    # create the args relevant for wandb
    
    

    device = torch.device(args.device)
    print(f'Using device: {device}')

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    if args.enable_dali:
        # this creates the DALI dataloaders
        data_loader_train, data_loader_val = get_data_loaders_dali(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            workers=args.workers,
            local_rank=args.gpu,
            world_size=args.world_size,
            dali_cpu=args.dali_cpu,
            crop_size=args.train_crop_size,
            val_size=args.val_resize_size,
        )
    else:
        # this creates the standard PyTorch data loaders using the Imagenet1k utility
        dataset_train, dataset_val, train_sampler, val_sampler = load_data(args)
        
        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=num_classes, use_v2=args.use_v2
        )
        if mixup_cutmix is not None:
            def collate_fn(batch):
                return mixup_cutmix(*default_collate(batch))

        else:
            collate_fn = default_collate

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.workers, pin_memory=True
        )

    num_classes = 1000
    
    # modified here to use custom model load operations!
    # apply linearization and apex here
    resnets = ['resnet18', 'resnet34', 'resnet50']
    vggs = ['vgg11', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']
    efficientnets = ['efficientnet_S']
    shufflenets = ['shuffleV2_x2_0']
    convnexts = ['convnext_T', 'convnext_S']
    known_models = [*resnets, *vggs, *shufflenets, *efficientnets, *convnexts]
    
    gumbel_2x4 = args.sparsity == 'gumbel_2x4'
    if not(args.model in known_models):
        print(f'Received argument model: {args.model}, which is not in {known_models}. No custom model load operations are provided.')   
        print(f'Defaulting to default behaviour of train.py.')
        model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    
    elif args.model in resnets:
        print(f'Loading model {args.model} with weights {args.weights} and sparsity {args.sparsity}.')
        if not os.path.exists(args.weights):
            raise ValueError(f'Weights file {args.weights} does not exist!')
        
        model_to_func = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50
        }
        sparse_model_to_func = {
            'resnet18': resnet18_sparse,
            'resnet34': resnet34_sparse,
            'resnet50': resnet50_sparse
        }
        
        if args.sparsity == 'none':
            model = model_to_func[args.model]()
            model.load_state_dict(torch.load(args.weights))
        elif args.sparsity in ['sparse_2x4', 'gumbel_2x4']:
            model = sparse_model_to_func[args.model](sparse_2x4=True, gumbel_2x4=gumbel_2x4, augment_2x4=args.aug_for_2x4, blocked=args.blocked_mmm, ignore_linear=not(args.override_ignore_linear))
            renamed_dict = rename_weights_resnet(torch.load(args.weights), blocked=args.blocked_mmm)
            model.load_state_dict(renamed_dict, strict=False)    
        else:
            raise ValueError("Unknown or unimplemented sparsity type.")
        
    elif args.model in vggs:
        print(f'Loading model {args.model} with weights {args.weights} and sparsity {args.sparsity}.')
        if not os.path.exists(args.weights):
            raise ValueError(f'Weights file {args.weights} does not exist!')
        
        model_to_func = {
            'vgg11': vgg11,
            'vgg11_bn': vgg11_bn,
            'vgg13_bn': vgg13_bn,
            'vgg16_bn': vgg16_bn,
            'vgg19_bn': vgg19_bn
        }
        sparse_model_to_func = {
            'vgg11': vgg11_sparse,
            'vgg11_bn': vgg11_bn_sparse,
            'vgg13_bn': vgg13_bn_sparse,
            'vgg16_bn': vgg16_bn_sparse,
            'vgg19_bn': vgg19_bn_sparse
        }
        
        if args.sparsity == 'none':
            model = model_to_func[args.model]()
            model.load_state_dict(torch.load(args.weights))
        elif args.sparsity in ['sparse_2x4', 'gumbel_2x4']:
            model = sparse_model_to_func[args.model](sparse_2x4=True, gumbel_2x4=gumbel_2x4, augment_2x4=args.aug_for_2x4, blocked=args.blocked_mmm, ignore_linear=not(args.override_ignore_linear))
            renamed_dict = rename_weights_vgg(torch.load(args.weights), blocked=args.blocked_mmm)
            model.load_state_dict(renamed_dict, strict=False)
        else:
            raise ValueError("Unknown or unimplemented linearization type.")

    elif args.model in shufflenets:
        print(f'Loading model {args.model} with weights {args.weights} and sparsity {args.sparsity}.')
        if not os.path.exists(args.weights):
            raise ValueError(f'Weights file {args.weights} does not exist!')

        model_to_func = {
            'shuffleV2_x2_0': shufflenet_v2_x2_0,
        }
        sparse_model_to_func = {
            'shuffleV2_x2_0': shufflenet_v2_x2_0_sparse,
        }
        
        orig_arch = model_to_func[args.model]()
        original_state = torch.load(args.weights)
        
        if args.sparsity == 'none':
            model = orig_arch
            model.load_state_dict(original_state)
        elif args.sparsity in ['sparse_2x4', 'gumbel_2x4']:
            sparse_arch = sparse_model_to_func[args.model](sparse_2x4=True, gumbel_2x4=gumbel_2x4, augment_2x4=args.aug_for_2x4, blocked=args.blocked_mmm)
            sparse_dict = rename_weights_shufflenet(original_state, original_arch=orig_arch, sparse_arch=sparse_arch, 
                                                      blocked=args.blocked_mmm, augment_2x4=(args.aug_for_2x4 and gumbel_2x4))
            model = sparse_arch
            model.load_state_dict(sparse_dict, strict=False)
    
    elif args.model == 'efficientnet_S':
        print(f'Loading model {args.model} with weights {args.weights} and sparsity {args.sparsity}.')
        if not os.path.exists(args.weights):
            raise ValueError(f'Weights file {args.weights} does not exist!')
        
        orig_arch = efficientnet_v2_s()
        original_state = torch.load(args.weights)
        
        if args.sparsity == 'none':
            model = orig_arch
            model.load_state_dict(original_state)
        elif args.sparsity in ['sparse_2x4', 'gumbel_2x4']:
            sparse_arch = efficientnet_v2_s_sparse(sparse_2x4=True, gumbel_2x4=gumbel_2x4, augment_2x4=args.aug_for_2x4, blocked=args.blocked_mmm)
            sparse_dict = rename_weights_efficientnet(original_state, original_arch=orig_arch, sparse_arch=sparse_arch, 
                                                      blocked=args.blocked_mmm, augment_2x4=(args.aug_for_2x4 and gumbel_2x4))
            model = sparse_arch
            model.load_state_dict(sparse_dict, strict=False)
    
    elif args.model in convnexts:
        print(f'Loading model {args.model} with weights {args.weights} and sparsity {args.sparsity}.')
        if not os.path.exists(args.weights):
            raise ValueError(f'Weights file {args.weights} does not exist!')

        model_to_func = {
            'convnext_T': convnext_tiny,
            'convnext_S': convnext_small,
        }
        sparse_model_to_func = {
            'convnext_T': convnext_tiny_sparse,
            'convnext_S': convnext_small_sparse,
        }
        
        orig_arch = model_to_func[args.model]()
        original_state = torch.load(args.weights)
        
        if args.sparsity == 'none':
            model = orig_arch
            model.load_state_dict(original_state)
        elif args.sparsity in ['sparse_2x4', 'gumbel_2x4']:
            orig_arch.load_state_dict(original_state)
            sparse_arch = sparse_model_to_func[args.model](sparse_2x4=True, gumbel_2x4=gumbel_2x4, augment_2x4=args.aug_for_2x4, blocked=args.blocked_mmm, ignore_grouped=args.ignore_grouped, ignore_linear=not(args.override_ignore_linear))
            sparse_dict = compute_sparse_state_dict_next(orig_arch, sparse_arch, augment_2x4=args.aug_for_2x4, ignore_grouped=args.ignore_grouped)
            model = sparse_arch
            model.load_state_dict(sparse_dict, strict=False)
            
    if args.trainable == 'gumbel_2x4' and args.sparsity == 'gumbel_2x4':
        trainables = ['choice_weights', 'masking_patterns']
        for name, param in model.named_parameters():
            if any([trainable in name for trainable in trainables]):
                param.requires_grad = True 
            else:
                param.requires_grad = False
    elif args.trainable == 'gumbel_2x4':
        print(f'Only gumbel_2x4 sparsity is supported for trainable gumbel_2x4 models. Defaulting to all trainable behaviour.')
    if args.trainable == 'none':
        for param in model.parameters():
            param.requires_grad = False
    
    if args.channels_last:
        print(f'Using channels last memory layout.')
        model = model.to(memory_format=torch.channels_last)
        
    if args.compile:
        print(f'Compiling model with torch.compile().')
        model = torch.compile(model)
    
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)
        
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    
    # modified here to use apex for sparse MMA
    # at this point the optimizer has been set and therefore it is possible to apply sparsity pruning
    if args.enable_apex:
        print(f'Enabling APEX for sparse MMA with permutation turned {"on" if args.apex_permutation else "off"}.')
        try:
            from apex.contrib.sparsity import ASP
        except:
            raise ImportError("Could not import APEX.")
        # the following line is the wanted behaviour, but does not allow for setting the permutation of channels
        # ASP.prune_trained_model(model, optimizer)
        # the following workaround is used instead, cf. https://github.com/NVIDIA/apex/blob/master/apex/contrib/sparsity/asp.py @ l. 292
        ASP.init_model_for_pruning(model, mask_calculator="m4n2_1d", verbosity=args.apex_verbosity, whitelist=[torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention], allow_recompute_mask=False, allow_permutation=args.apex_permutation)
        ASP.init_optimizer_for_pruning(optimizer)
        ASP.compute_sparse_masks()
        model.to(device)
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        model.to(device)

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    # create a global logger to not interfere with wandb
    # TODO: pass wandb args!!
    wandb_args = {
        '_wandb_architecture': args.model,
        '_wandb_dataset': 'Imagenet-1k',
        '_wandb_sparsity' : args.sparsity,
        '_wandb_weights' : args.weights,
        '_wandb_apex' : args.enable_apex,
        '_wandb_apex_permutation' : args.apex_permutation,
    }
    metric_logger = utils.MetricLogger(delimiter="  ", wandb=args.use_wandb, **wandb_args)
    if not args.test_only:
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_val, device=device, log_suffix="EMA", global_metric_logger=metric_logger)
        else:
            evaluate(model, criterion, data_loader_val, device=device, global_metric_logger=metric_logger)
        return


    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and not args.enable_dali:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader_train, device, epoch, args, model_ema, scaler, global_metric_logger=metric_logger)
        lr_scheduler.step()
        evaluate(model, criterion, data_loader_val, device=device, global_metric_logger=metric_logger)
        if model_ema:
            evaluate(model_ema, criterion, data_loader_val, device=device, log_suffix="EMA", global_metric_logger=metric_logger)
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training (fp16). (default: False)")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load/the weights file path to load from")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    
    # own arguments go here
    parser.add_argument("--use-wandb", default=False, action='store_true', help='Use Weights and Biases for logging. (default: False)')
    parser.add_argument("--wandb-project", default='practical_2', type=str, help='Weights and Biases project name. (defaults)')
    parser.add_argument("--wandb-log-dir", default='/itet-stor/ddanhofer/net_scratch/logs_2/wandb', type=str, help='Weights and Biases log directory. (defaults)')
    
    parser.add_argument('--sparsity', choices=['none', 'sparse_2x4', 'gumbel_2x4'], default='none', type=str, help='The linearization strategy to use. Can be: none, lin, or gumbel_24. (default: none)')
    parser.add_argument('--aug-for-2x4', dest='aug_for_2x4', default=False, action='store_true', help='Enable augmentation for models non "natively" 2x4-compatible. (default: False)')
    parser.add_argument('--blocked-mmm', dest='blocked_mmm', default=False, action='store_true', help='Enable blocked MMM for sparse matrices. (default: False)')
    parser.add_argument('--enable-apex', dest='enable_apex', default=False, action='store_true', help='Enable APEX for sparse MMA. (default: False)')        
    parser.add_argument('--apex-verbosity', default=2, type=int, choices=[0, 1, 2, 3], help='Set APEX verbosity. (default: 2)')        
    parser.add_argument('--apex-permutation', default=False, action='store_true', help='Enable APEX channel permutation for weight pruning. (default: False)')        
    parser.add_argument('--ignore-grouped', default=False, action='store_true', help='Ignore grouped convolutions for sparsity. (default: False)')        
    parser.add_argument('--override-ignore-linear', default=False, action='store_true', help='Override the default of ignoring linear layers for sparsity. (default: False)')        
    
    parser.add_argument('--trainable', choices=['all', 'gumbel_2x4', 'none'], default='all', type=str, help='Set the trainable parameters of the model. Can be: all, gumbel_2x4, or none. (default: all)')
    parser.add_argument('--compile', default=False, action='store_true', help='Compile the model with torch.compile(_). (default: False)')
    parser.add_argument('--channels-last', default=False, action='store_true', help='Use channels last memory layout. (default: False)')
    
    parser.add_argument('--enable-dali', default=False, action='store_true',
                        help='Enable DALI data loader instead of native PyTorch DataLoader. (default: False)')
    parser.add_argument('--dali-cpu', action='store_true', default=False,
                        help='Runs CPU based version of DALI pipeline. (default: False)')
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
