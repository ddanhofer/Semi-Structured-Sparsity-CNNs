# Inducing Semi-Structured Sparsity by Masking for Efficient Model Inference in Convolutional Networks

## Project Overview

This repository contains the code to reproduce the numbers reported in the paper [Inducing Semi-Structured Sparsity by Masking for Efficient Model Inference in Convolutional Networks](https://arxiv.org/pdf/2411.00288). It contains the proposed architectural change to obtain 2:4 sparse convolutional layers, the original and modified architectures, data loading utilites, the training script and the execution shell script. Some assets provided in the repository are derived from previously existing and licensed assets. Refer to the source code to view details.

---

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
  - [Modified Convolutional Layers](#modified-convolutional-layers)
  - [ResNet](#resnet)
  - [ConvNeXt](#convnext)
  - [Dataset Utilities](#dataset-utilities)
- [Running Experiments](#running-experiments)
- [Training Script](#training-script)
- [License](#license)

---

## Installation

To set up the execution environment containing the required dependencies to reproduce the experiments it is recommended to use `conda` to set up a dedicated environment and install the requirements from the provided `requirements.txt` file.

---

## Project Structure

The project contains several components, including models, modified convolutional layers, dataset utilities, and scripts for training and running experiments as well as other utilities.

### Modified Convolutional Layers

- The `linearization/` folder contains the 2:4 sparse convolutional layers proposed in the paper and used for modifying the models.

### ResNet

- The **original ResNet** model and
- The **modified ResNet** (with custom convolutional layers) are stored in the `resnet/` folder.

### ConvNeXt

- The **original ConvNeXt** model and
- The **modified ConvNeXt** (with custom convolutional layers) are stored in the `convnext/` folder.

### Dataset Utilities

- Dataset loading utilities are located in the `/data_set/` directory. This directory contains the dataset loaders used with and without DALI enabled.

---

## Running Experiments

A shell script is provided in the root directory to run experiments. Before execution the script needs to be modified by providing the values of three variables pointing to the output directory, the data directory and the location of the weights 

## Training Script

The training script `train_sparse.py` is located in the `classification` folder and can be used to train and evaluate the dense and sparse models. Below is the help message for the training script for use beyond the provided commands to reproduce experiments:

```
usage: train_sparse.py [-h] [--data-path DATA_PATH] [--model MODEL] [--device DEVICE] [-b BATCH_SIZE] [--epochs N] [-j N]
                       [--opt OPT] [--lr LR] [--momentum M] [--wd W] [--norm-weight-decay NORM_WEIGHT_DECAY]
                       [--bias-weight-decay BIAS_WEIGHT_DECAY] [--transformer-embedding-decay TRANSFORMER_EMBEDDING_DECAY]
                       [--label-smoothing LABEL_SMOOTHING] [--mixup-alpha MIXUP_ALPHA] [--cutmix-alpha CUTMIX_ALPHA]
                       [--lr-scheduler LR_SCHEDULER] [--lr-warmup-epochs LR_WARMUP_EPOCHS] [--lr-warmup-method LR_WARMUP_METHOD]
                       [--lr-warmup-decay LR_WARMUP_DECAY] [--lr-step-size LR_STEP_SIZE] [--lr-gamma LR_GAMMA] [--lr-min LR_MIN]
                       [--print-freq PRINT_FREQ] [--output-dir OUTPUT_DIR] [--resume RESUME] [--start-epoch N] [--cache-dataset]
                       [--sync-bn] [--test-only] [--auto-augment AUTO_AUGMENT] [--ra-magnitude RA_MAGNITUDE]
                       [--augmix-severity AUGMIX_SEVERITY] [--random-erase RANDOM_ERASE] [--amp] [--world-size WORLD_SIZE]
                       [--dist-url DIST_URL] [--model-ema] [--model-ema-steps MODEL_EMA_STEPS] [--model-ema-decay MODEL_EMA_DECAY]
                       [--use-deterministic-algorithms] [--interpolation INTERPOLATION] [--val-resize-size VAL_RESIZE_SIZE]
                       [--val-crop-size VAL_CROP_SIZE] [--train-crop-size TRAIN_CROP_SIZE] [--clip-grad-norm CLIP_GRAD_NORM]
                       [--ra-sampler] [--ra-reps RA_REPS] [--weights WEIGHTS] [--backend BACKEND] [--use-v2] [--use-wandb]
                       [--wandb-project WANDB_PROJECT] [--wandb-log-dir WANDB_LOG_DIR] [--sparsity {none,sparse_2x4,gumbel_2x4}]
                       [--aug-for-2x4] [--blocked-mmm] [--enable-apex] [--apex-verbosity {0,1,2,3}] [--apex-permutation]
                       [--ignore-grouped] [--override-ignore-linear] [--trainable {all,gumbel_2x4,none}] [--compile]
                       [--channels-last] [--enable-dali] [--dali-cpu]

PyTorch Classification Training

options:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        dataset path
  --model MODEL         model name
  --device DEVICE       device (Use cuda or cpu Default: cuda)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        images per gpu, the total batch size is $NGPU x batch_size
  --epochs N            number of total epochs to run
  -j N, --workers N     number of data loading workers (default: 16)
  --opt OPT             optimizer
  --lr LR               initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  --norm-weight-decay NORM_WEIGHT_DECAY
                        weight decay for Normalization layers (default: None, same value as --wd)
  --bias-weight-decay BIAS_WEIGHT_DECAY
                        weight decay for bias parameters of all layers (default: None, same value as --wd)
  --transformer-embedding-decay TRANSFORMER_EMBEDDING_DECAY
                        weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)
  --label-smoothing LABEL_SMOOTHING
                        label smoothing (default: 0.0)
  --mixup-alpha MIXUP_ALPHA
                        mixup alpha (default: 0.0)
  --cutmix-alpha CUTMIX_ALPHA
                        cutmix alpha (default: 0.0)
  --lr-scheduler LR_SCHEDULER
                        the lr scheduler (default: steplr)
  --lr-warmup-epochs LR_WARMUP_EPOCHS
                        the number of epochs to warmup (default: 0)
  --lr-warmup-method LR_WARMUP_METHOD
                        the warmup method (default: constant)
  --lr-warmup-decay LR_WARMUP_DECAY
                        the decay for lr
  --lr-step-size LR_STEP_SIZE
                        decrease lr every step-size epochs
  --lr-gamma LR_GAMMA   decrease lr by a factor of lr-gamma
  --lr-min LR_MIN       minimum lr of lr schedule (default: 0.0)
  --print-freq PRINT_FREQ
                        print frequency
  --output-dir OUTPUT_DIR
                        path to save outputs
  --resume RESUME       path of checkpoint
  --start-epoch N       start epoch
  --cache-dataset       Cache the datasets for quicker initialization. It also serializes the transforms
  --sync-bn             Use sync batch norm
  --test-only           Only test the model
  --auto-augment AUTO_AUGMENT
                        auto augment policy (default: None)
  --ra-magnitude RA_MAGNITUDE
                        magnitude of auto augment policy
  --augmix-severity AUGMIX_SEVERITY
                        severity of augmix policy
  --random-erase RANDOM_ERASE
                        random erasing probability (default: 0.0)
  --amp                 Use torch.cuda.amp for mixed precision training (fp16). (default: False)
  --world-size WORLD_SIZE
                        number of distributed processes
  --dist-url DIST_URL   url used to set up distributed training
  --model-ema           enable tracking Exponential Moving Average of model parameters
  --model-ema-steps MODEL_EMA_STEPS
                        the number of iterations that controls how often to update the EMA model (default: 32)
  --model-ema-decay MODEL_EMA_DECAY
                        decay factor for Exponential Moving Average of model parameters (default: 0.99998)
  --use-deterministic-algorithms
                        Forces the use of deterministic algorithms only.
  --interpolation INTERPOLATION
                        the interpolation method (default: bilinear)
  --val-resize-size VAL_RESIZE_SIZE
                        the resize size used for validation (default: 256)
  --val-crop-size VAL_CROP_SIZE
                        the central crop size used for validation (default: 224)
  --train-crop-size TRAIN_CROP_SIZE
                        the random crop size used for training (default: 224)
  --clip-grad-norm CLIP_GRAD_NORM
                        the maximum gradient norm (default None)
  --ra-sampler          whether to use Repeated Augmentation in training
  --ra-reps RA_REPS     number of repetitions for Repeated Augmentation (default: 3)
  --weights WEIGHTS     the weights enum name to load/the weights file path to load from
  --backend BACKEND     PIL or tensor - case insensitive
  --use-v2              Use V2 transforms
  --use-wandb           Use Weights and Biases for logging. (default: False)
  --wandb-project WANDB_PROJECT
                        Weights and Biases project name. (defaults)
  --wandb-log-dir WANDB_LOG_DIR
                        Weights and Biases log directory. (defaults)
  --sparsity {none,sparse_2x4,gumbel_2x4}
                        The linearization strategy to use. Can be: none, lin, or gumbel_24. (default: none)
  --aug-for-2x4         Enable augmentation for models non "natively" 2x4-compatible. (default: False)
  --blocked-mmm         Enable blocked MMM for sparse matrices. (default: False)
  --enable-apex         Enable APEX for sparse MMA. (default: False)
  --apex-verbosity {0,1,2,3}
                        Set APEX verbosity. (default: 2)
  --apex-permutation    Enable APEX channel permutation for weight pruning. (default: False)
  --ignore-grouped      Ignore grouped convolutions for sparsity. (default: False)
  --override-ignore-linear
                        Override the default of ignoring linear layers for sparsity. (default: False)
  --trainable {all,gumbel_2x4,none}
                        Set the trainable parameters of the model. Can be: all, gumbel_2x4, or none. (default: all)
  --compile             Compile the model with torch.compile(_). (default: False)
  --channels-last       Use channels last memory layout. (default: False)
  --enable-dali         Enable DALI data loader instead of native PyTorch DataLoader. (default: False)
  --dali-cpu            Runs CPU based version of DALI pipeline. (default: False)
```

---

## License

This project is licensed under the Creative Commons License - see the [LICENSE](LICENSE) file for details.

---

If you encounter any issues or have questions, feel free to open an issue.
