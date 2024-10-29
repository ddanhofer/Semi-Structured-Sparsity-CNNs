import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torchvision.transforms as transforms
from typing import Callable

class Imagenet1k(Dataset):
    def __init__(self, split='train', transform: Callable = lambda x: x):
        dataset = load_dataset('imagenet-1k', cache_dir='/itet-stor/ddanhofer/net_scratch/data/')
        subset = dataset[split]
        
        self.transform = transform
        self.dataset = subset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.transform(sample['image'])
        return image, torch.tensor(sample['label'])

if __name__ == '__main__':
    import sys
    sys.path.insert(0, './')
    
    from torchvision.transforms.functional import InterpolationMode
    import time
    import classification.presets as presets

    preprocessing = presets.ClassificationPresetEval(
                crop_size=224,
                resize_size=256,
                interpolation=InterpolationMode('bilinear'),
                backend='PIL',
                use_v2=True,
            )
    
    BATCH_SIZE = 192
    train_dataset = Imagenet1k(split='train', transform=preprocessing)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    

    for k in [1, 3, 5, 10, 20, 50, 100]:
        start_time = time.time()
        ctr = 0
        
        for i, (images, labels) in enumerate(train_loader):
            ctr += 1
            if ctr > k:
                ctr = 0
                break

        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f'{k}its == {k * BATCH_SIZE} imgs: {execution_time} s => {k * BATCH_SIZE / execution_time} img/s')
