# this script contains the commands for all figures reported in the paper and not drawn from a previous publication
# the commands depend on three directories:
# (1) path_to_weights_dir: the directory containing the pre-trained weights for the models
# (2) path_to_data_dir: the directory containing the ImageNet data
# (3) path_to_output_dir: the directory where the training output, i.e., checkpoints of the models, will be stored
# additionally the use of Weights & Biases for logging can be enabled by adding the flag --use-wandb
# the use of DALI for data loading can be enabled by adding the flag --enable-dali
# for more information on the use of train_sparse.py also see the help message of the script: python classification/train_sparse.py --help

# The output of the commands is piped to stdout/stderr and can be redirected to a file for logging purposes


# Table 1: Validation classification performance of the 2:4 sparse networks measured as the top-k
# accuracy on ImageNet-1K [14] and the number of epochs needed to converge to a comparable or
# better top-1 accuracy than reported in contrast to the original number of training epochs
    # ResNets: ResNet-18, ResNet-34, ResNet-50
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 --nproc_per_node=1 classification/train_sparse.py --opt adamw --lr 1.0 --lr-step-size 3 --world-size 1 --weights [path_to_weights_dir]/resnet/resnet18-f37072fd.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/resnet18_2x4 --model resnet18 --workers 16 [--use-wandb] [--enable-dali] --sparsity gumbel_2x4 --trainable gumbel_2x4 --batch-size 256 --use-v2 --epochs 1
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29501 --nproc_per_node=1 classification/train_sparse.py --opt adamw --lr 1.0 --lr-step-size 3 --world-size 1 --weights [path_to_weights_dir]/resnet/resnet34-b627a593.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/resnet34_2x4 --model resnet34 --workers 16 [--use-wandb] [--enable-dali] --sparsity gumbel_2x4 --trainable gumbel_2x4 --batch-size 256 --use-v2 --epochs 1
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29502 --nproc_per_node=1 classification/train_sparse.py --opt adamw --lr 1.0 --lr-step-size 3 --world-size 1 --weights [path_to_weights_dir]/resnet/resnet50-0676ba61.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/resnet50_2x4 --model resnet50 --workers 16 [--use-wandb] [--enable-dali] --sparsity gumbel_2x4 --trainable gumbel_2x4 --batch-size 128 --use-v2 --epochs 1
    # ConvNeXts: ConvNeXt-T, ConvNeXt-S
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29503 --nproc_per_node=1 classification/train_sparse.py --opt adamw --lr 1.0 --lr-step-size 3 --world-size 1 --weights [path_to_weights_dir]/convnext/convnext_tiny-983f1562.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/next_T_2x4 --model convnext_T --workers 16 [--use-wandb] [--enable-dali] --sparsity gumbel_2x4 --trainable gumbel_2x4 --batch-size 32 --use-v2 --blocked-mmm --aug-for-2x4 --ignore-grouped  --epochs 10
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29504 --nproc_per_node=1 classification/train_sparse.py --opt adamw --lr 1.0 --lr-step-size 3 --world-size 1 --weights [path_to_weights_dir]/convnext/convnext_small-0c510722.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/next_S_2x4 --model convnext_S --workers 16 [--use-wandb] [--enable-dali] --sparsity gumbel_2x4 --trainable gumbel_2x4 --batch-size 32 --use-v2 --blocked-mmm --aug-for-2x4 --ignore-grouped --epochs 9


# Table 2: Validation classification performance of the 2:4 sparse networks measured as the top-k
# accuracy on ImageNet [14] after spending 10% of the resources used to initially train the network
# measured by the number of epochs spent
    # ResNets: ResNet-18, ResNet-34, ResNet-50
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29600 --nproc_per_node=1 classification/train_sparse.py --opt adamw --lr 1.0 --lr-step-size 3 --world-size 1 --weights [path_to_weights_dir]/resnet/resnet18-f37072fd.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/resnet18_2x4 --model resnet18 --workers 16 [--use-wandb] [--enable-dali] --sparsity gumbel_2x4 --trainable gumbel_2x4 --batch-size 256 --use-v2 --epochs 9
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29601 --nproc_per_node=1 classification/train_sparse.py --opt adamw --lr 1.0 --lr-step-size 3 --world-size 1 --weights [path_to_weights_dir]/resnet/resnet34-b627a593.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/resnet34_2x4 --model resnet34 --workers 16 [--use-wandb] [--enable-dali] --sparsity gumbel_2x4 --trainable gumbel_2x4 --batch-size 256 --use-v2 --epochs 9
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29602 --nproc_per_node=1 classification/train_sparse.py --opt adamw --lr 1.0 --lr-step-size 3 --world-size 1 --weights [path_to_weights_dir]/resnet/resnet50-0676ba61.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/resnet50_2x4 --model resnet50 --workers 16 [--use-wandb] [--enable-dali] --sparsity gumbel_2x4 --trainable gumbel_2x4 --batch-size 128 --use-v2 --epochs 9
    # ConvNeXts: ConvNeXt-T, ConvNeXt-S
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29603 --nproc_per_node=1 classification/train_sparse.py --opt adamw --lr 1.0 --lr-step-size 3 --world-size 1 --weights [path_to_weights_dir]/convnext/convnext_tiny-983f1562.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/next_T_2x4 --model convnext_T --workers 16 [--use-wandb] [--enable-dali] --sparsity gumbel_2x4 --trainable gumbel_2x4 --batch-size 32 --use-v2 --blocked-mmm --aug-for-2x4 --ignore-grouped --epochs 30
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29604 --nproc_per_node=1 classification/train_sparse.py --opt adamw --lr 1.0 --lr-step-size 3 --world-size 1 --weights [path_to_weights_dir]/convnext/convnext_small-0c510722.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/next_S_2x4 --model convnext_S --workers 16 [--use-wandb] [--enable-dali] --sparsity gumbel_2x4 --trainable gumbel_2x4 --batch-size 32 --use-v2 --blocked-mmm --aug-for-2x4 --ignore-grouped --epochs 30

# Table 3: Validation classification performance of the 2:4 sparse networks obtained via the apex library
# [13] measured as the top-k accuracy on ImageNet [14]. The networks are compared in two settings
# disallowing and allowing permutations of the channels before pruning.
    # no channel permutation
        # ResNets: ResNet-18, ResNet-34, ResNet-50
        torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29700 --nproc_per_node=1 classification/train_sparse.py --world-size 1 --weights [path_to_weights_dir]/resnet/resnet18-f37072fd.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/resnet18_ampn --model resnet18 --workers 16 [--use-wandb] [--enable-dali] --sparsity none --trainable all --batch-size 256 --use-v2 --test-only --enable-apex
        torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29701 --nproc_per_node=1 classification/train_sparse.py --world-size 1 --weights [path_to_weights_dir]/resnet/resnet34-b627a593.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/resnet34_ampn --model resnet34 --workers 16 [--use-wandb] [--enable-dali] --sparsity none --trainable all --batch-size 256 --use-v2 --test-only --enable-apex
        torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29702 --nproc_per_node=1 classification/train_sparse.py --world-size 1 --weights [path_to_weights_dir]/resnet/resnet50-0676ba61.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/resnet50_ampn --model resnet50 --workers 16 [--use-wandb] [--enable-dali] --sparsity none --trainable all --batch-size 256 --use-v2 --test-only --enable-apex
        # ConvNeXts: ConvNeXt-T, ConvNeXt-S
    # channel permutation
        # ResNets: ResNet-18, ResNet-34, ResNet-50
        torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29700 --nproc_per_node=1 classification/train_sparse.py --world-size 1 --weights [path_to_weights_dir]/resnet/resnet18-f37072fd.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/resnet18_amp --model resnet18 --workers 16 [--use-wandb] [--enable-dali] --sparsity none --trainable all --batch-size 256 --use-v2 --test-only --enable-apex --apex-permutation
        torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29800 --nproc_per_node=1 classification/train_sparse.py --world-size 1 --weights [path_to_weights_dir]/resnet/resnet34-b627a593.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/resnet34_amp --model resnet34 --workers 16 [--use-wandb] [--enable-dali] --sparsity none --trainable all --batch-size 256 --use-v2 --test-only --enable-apex --apex-permutation
        torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29800 --nproc_per_node=1 classification/train_sparse.py --world-size 1 --weights [path_to_weights_dir]/resnet/resnet50-0676ba61.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/resnet50_amp --model resnet50 --workers 16 [--use-wandb] [--enable-dali] --sparsity none --trainable all --batch-size 256 --use-v2 --test-only --enable-apex --apex-permutation



# Table 4: Reported and validation classification performance of the unmodified architectures measured
# as the top-k accuracy on ImageNet [14]
    # ResNets: ResNet-18, ResNet-34, ResNet-50 
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29700 --nproc_per_node=1 classification/train_sparse.py --world-size 1 --weights [path_to_weights_dir]/resnet/resnet18-f37072fd.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/resnet18_none --model resnet18 --workers 16 [--use-wandb] [--enable-dali] --sparsity none --trainable all --batch-size 256 --use-v2 --test-only
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29701 --nproc_per_node=1 classification/train_sparse.py --world-size 1 --weights [path_to_weights_dir]/resnet/resnet34-b627a593.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/resnet34_none --model resnet34 --workers 16 [--use-wandb] [--enable-dali] --sparsity none --trainable all --batch-size 192 --use-v2 --test-only
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29702 --nproc_per_node=1 classification/train_sparse.py --world-size 1 --weights [path_to_weights_dir]/resnet/resnet50-0676ba61.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/resnet50_none --model resnet50 --workers 16 [--use-wandb] [--enable-dali] --sparsity none --trainable all --batch-size 192 --use-v2 --test-only
    # ConvNeXts: ConvNeXt-T, ConvNeXt-S
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29703 --nproc_per_node=1 classification/train_sparse.py --world-size 1 --weights [path_to_weights_dir]/convnext/convnext_tiny-983f1562.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/next_T_none --model convnext_T --workers 16 [--use-wandb] [--enable-dali] --sparsity none --trainable all --batch-size 32 --use-v2 --test-only
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29704 --nproc_per_node=1 classification/train_sparse.py --world-size 1 --weights [path_to_weights_dir]/convnext/convnext_small-0c510722.pth --data-path [path_to_data_dir]/data/ --output-dir [path_to_output_dir]/output/next_S_none --model convnext_S --workers 16 [--use-wandb] [--enable-dali] --sparsity none --trainable all --batch-size 32 --use-v2 --test-only
