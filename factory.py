import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from torch.utils.data import Dataset, DataLoader, default_collate
from typing import List, Iterable, Optional
import os
import cv2
from torchvision.transforms import v2
from typing import Dict
import yaml

from model import EfficientNet, ConvNeXt
from resnet import ResNet
from simple import SimpleCNN, SimpleCNNv2
from vision_transformer import VisionTransformer

def model_factory(model_type: str, **kwargs) -> nn.Module:
    models = {
        'convnext' : ConvNeXt,
        'efficientnet' : EfficientNet,
        'resnet' : ResNet,
        'simplecnn' : SimpleCNN,
        'simplecnnv2' : SimpleCNNv2,
        'vision_transformer' : VisionTransformer
    }

    if model_type not in models:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type](**kwargs)

def optimizer_factory(optimizer_type: str, parameters: Iterable[nn.Parameter], **kwargs) -> Optimizer:
    optimizers = {
        'sgd' : SGD,
        'adam' : Adam,
        'adamw' : AdamW,
    }

    if optimizer_type not in optimizers:
        raise ValueError(f'Invalid optimizer_type: {optimizer_type}. Choose from {list(optimizers.keys())}.')
    
    return optimizers[optimizer_type](parameters, **kwargs)

def lr_scheduler_factory(optimizer: Optimizer, scheduler_configs: List) -> lr_scheduler.LRScheduler:
    schedulers = {
        'cosine_annealing' : lr_scheduler.CosineAnnealingLR,
        'cosine_annealing_warm_restarts' : lr_scheduler.CosineAnnealingWarmRestarts,
        'const' : lr_scheduler.ConstantLR,
        'exp' : lr_scheduler.ExponentialLR,
        'linear' : lr_scheduler.LinearLR,
        'step' : lr_scheduler.StepLR,
    }

    list_of_schedulers = []
    milestones = []
    for scheduler_config in scheduler_configs:
        scheduler = scheduler_config.pop('type', None)
        if scheduler not in schedulers:
            raise ValueError(f'Invalid scheduler type: {scheduler}. Choose from {list(schedulers.keys())}.')
        epoch = scheduler_config.pop('epoch', None)
        if epoch is not None:
            milestones.append(epoch)
        list_of_schedulers.append(schedulers[scheduler](optimizer, **scheduler_config))

    if len(milestones) != len(list_of_schedulers) - 1:
        raise ValueError("The number of milestones should be exactly one less than the number of schedulers.")

    return lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=list_of_schedulers, milestones=milestones)

def pre_augmentation_factory(augmentations: List[Dict]) -> v2.Transform:
    pre_augmentations = {
        'rotation' : v2.RandomRotation,
        'horizontal_flip' : v2.RandomHorizontalFlip,
        'randaugment' : v2.RandAugment,
    }

    used_augmentations = []

    for augmentation in augmentations:
        augmentation_type = augmentation.pop('type')
        used_augmentations.append(pre_augmentations[augmentation_type](**augmentation))

    return v2.Compose(transforms=used_augmentations)

def post_augmentation_factory(num_classes: int, augmentations: List[Dict]) -> v2.Transform:
    post_augmentations = {
        'cutmix' : v2.CutMix,
        'mixup' : v2.MixUp
    }

    used_augmentations = []
    p = []

    for augmentation in augmentations:
        augmentation_type = augmentation.pop('type')
        probability = augmentation.pop('p')
        used_augmentations.append(post_augmentations[augmentation_type](num_classes=num_classes, **augmentation))
        p.append(probability)

    return v2.RandomChoice(transforms=used_augmentations, p=p)

class ClassificationDataset(Dataset):
    def __init__(self, n_classes: int, path: str, transforms: Optional[v2.Transform] = None):
        self.path = path
        self.n_classes = n_classes
        self.filenames = os.listdir(self.path)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        loc = os.path.join(self.path, self.filenames[index])
        img = cv2.imread(loc)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.filenames[index].split('_')[-1].split('.')[0]
        label = int(label)

        if self.transforms is not None:
            img = self.transforms(img)

        #label = F.one_hot(torch.tensor(label), num_classes=self.n_classes).float()

        return img, label
    
def dataloader_factory(dataset_config: Dict, subset: str, transforms: Optional[v2.Transform] = None) -> DataLoader:
    n_classes = dataset_config.get('n_classes', 1000)
    batch_size = dataset_config.get('batch_size', 128)
    num_workers = dataset_config.get('num_workers', 24)

    dataset_config = dataset_config.get(subset, '')
    dataset_path = dataset_config.get('path', './')

    dataset = ClassificationDataset(
        n_classes=n_classes,
        path=dataset_path,
        transforms=transforms,
    )

    post_augmentations_config = dataset_config.get('post_augmentations', None)
    
    if post_augmentations_config is not None:
        post_augmentations = post_augmentation_factory(num_classes=n_classes, augmentations=post_augmentations_config)
        def collate_fn(batch) -> torch.Tensor:
            return post_augmentations(*default_collate(batch))
    else:
        def collate_fn(batch) -> torch.Tensor:
            return default_collate(batch)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True if subset == 'train' else False,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    return dataloader

if __name__ == '__main__':
    with open('config.yaml') as stream:
        config = yaml.safe_load(stream)
    dataset_config = config.get('dataset')
    dataloader = dataloader_factory(dataset_config=dataset_config, subset='val')