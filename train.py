import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import argparse
import yaml
import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from factory import model_factory, optimizer_factory, lr_scheduler_factory, dataloader_factory, pre_augmentation_factory
from transformer_engine.common.recipe import DelayedScaling, Format
import transformer_engine.pytorch as te
from contextlib import nullcontext
import math
import torch.backends.cudnn as cudnn
#import torch_tensorrt
import warnings
#from copy import deepcopy
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

os.environ['NVTE_FLASH_ATTN'] = '1'
os.environ['NVTE_FUSED_ATTN'] = '1'

warnings.filterwarnings('ignore')
cudnn.benchmark = True

def load_config() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_dir = args.config
    with open(config_dir, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def get_optimizer(model: nn.Module, train_config: Dict) -> optim.Optimizer:
    optimizer_config = train_config.get('optimizer')
    optimizer = optimizer_factory(**optimizer_config, parameters=model.parameters()) #type: ignore
    return optimizer

def get_scheduler(optimizer: optim.Optimizer, train_config: Dict) -> optim.lr_scheduler.LRScheduler:
    scheduler_config = train_config.get('scheduler', dict())
    scheduler = lr_scheduler_factory(optimizer=optimizer, scheduler_configs=scheduler_config)
    return scheduler

def get_transforms(subset: str, dataset_config: Dict) -> v2.Transform:
    crop_resolution = dataset_config.get('crop_resolution', 224)
    pre_augmentations = dataset_config.get(subset, dict()).get('pre_augmentations', None)
    if subset == 'train':
        return v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomCrop(size=crop_resolution),
            pre_augmentation_factory(augmentations=pre_augmentations) if pre_augmentations is not None else v2.Identity()
        ])
    else:
        return v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.CenterCrop(size=crop_resolution),
            pre_augmentation_factory(augmentations=pre_augmentations) if pre_augmentations is not None else v2.Identity()
        ])
    
class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: optim.Optimizer, 
                 train_dataloader: DataLoader, 
                 val_dataloader: DataLoader,
                 criterion: nn.Module,
                 scheduler: optim.lr_scheduler.LRScheduler,
                 train_config: Dict) -> None:
        
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.scheduler = scheduler

        self.train_config = train_config
        self.epochs = self.train_config.get('epochs', 5)
        self.save_directory = self.train_config.get('save_directory', './')
        self.precision = self.train_config.get('precision', 'fp32')
        self.verbosity = self.train_config.get('verbosity', 2)
        self.ema_beta = self.train_config.get('ema_beta', None)
        self.ema_step = self.train_config.get('ema_step', 32)

        if self.precision in ['fp16', 'fp8']:
            self.scaler = torch.amp.GradScaler(device='cuda') #type: ignore
        if self.precision == 'fp8':
            self.fp8_recipe = DelayedScaling(
                fp8_format=Format.HYBRID,
                amax_history_len=16,
                amax_compute_algo='max'
            )

        if self.ema_beta is not None:
            self.ema = AveragedModel(model=self.model, device=torch.device('cuda'), multi_avg_fn=get_ema_multi_avg_fn(self.ema_beta))
            self.ema.eval()
        else:
            self.ema = None

    def _update_params(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm.tqdm(self.train_dataloader, desc=f'[{epoch+1}/{self.epochs}] Train') if self.verbosity >= 2 else self.train_dataloader
        for i, (images, labels) in enumerate(pbar):    
            images, labels = images.cuda(), labels.cuda()

            self.optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16) if self.precision != 'fp32' else nullcontext():
                with te.fp8_autocast(fp8_recipe=self.fp8_recipe) if self.precision == 'fp8' else nullcontext():
                    y_hat = self.model(images)
                    loss = self.criterion(y_hat, labels)
            if self.precision == 'fp32':
                loss.backward()
                self.optimizer.step()
            else:
                try:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                except Exception as e:
                    print(f'[{epoch}] Error {e} occurred during gradinet updates.')
                    self.optimizer.zero_grad()

            if self.ema is not None and i % self.ema_step == 1:
                self.ema.update_parameters(self.model)

            gt = labels if labels.dim() == 1 else labels.argmax(dim=-1)
            prediction = y_hat.argmax(dim=-1)
            total_loss += loss.item()
            total += labels.size(0)
            correct += (prediction == gt).sum().item()

            if type(pbar) is tqdm.tqdm:
                pbar.set_postfix_str(f'Loss: {loss}')

        avg_loss = total_loss / len(self.train_dataloader)
        acc = correct * 100 / total

        return avg_loss, acc


    def _validate(self, epoch: int, model_to_validate: str) -> Tuple[float, float]:
        if model_to_validate != 'ema':
            self.model.eval()
            model = self.model
        else:
            if self.ema is None:
                raise ValueError('Cannot validate EMA if there is no EMA.')
            model = self.ema
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar_desc = f'[{epoch+1}/{self.epochs} Validation]' if model_to_validate != 'ema' else f'[{epoch+1}/{self.epochs}] EMA Validation'
            pbar = tqdm.tqdm(self.val_dataloader, desc=pbar_desc) if self.verbosity >= 2 else self.val_dataloader
            for images, labels in pbar:
                images, labels = images.cuda(), labels.cuda()
                
                with torch.autocast(device_type='cuda', dtype=torch.float16) if self.precision != 'fp32' else nullcontext():
                    with te.fp8_autocast(fp8_recipe=self.fp8_recipe) if self.precision == 'fp8' else nullcontext():
                        y_hat = model(images)
                        loss = self.criterion(y_hat, labels)
                
                gt = labels if labels.dim() == 1 else labels.argmax(dim=-1)
                prediction = y_hat.argmax(dim=-1)
                total_loss += loss.item()
                total += labels.size(0)
                correct += (prediction == gt).sum().item()
                
                if type(pbar) is tqdm.tqdm:
                    pbar.set_postfix_str(f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(self.val_dataloader)
        acc = correct * 100 / total
        
        return avg_loss, acc
    
    def _plot(self, train_accs: List[float], val_accs: List[float], learning_rates: List[float], epoch: int, ema_accs: Optional[List[float]] = None):
        plt.style.use('dark_background')
        epochs_range = range(1, epoch+1)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        line1, = ax1.plot(epochs_range, train_accs, color='#fda29d', label='Train Accuracy')
        line2, = ax1.plot(epochs_range, val_accs, color='#ff63a1', label='Validation Accuracy')
        line3, = ax2.plot(epochs_range, learning_rates, color='#fcc39b', label='log10 Learning Rate')
        if ema_accs is not None:
            line4, = ax1.plot(epochs_range, ema_accs, color='#fe829f', label='EMA Accuracy')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy(%)')
        ax2.set_ylabel('log10 Learning Rate')
        plt.title('Training Metrics by Epoch')

        lines = [line3, line1, line2] if ema_accs is None else [line3, line1, line4, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left') #type: ignore

        ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

        fig.tight_layout()

        plt.savefig(os.path.join(self.save_directory, 'plot.png'))

        plt.close(fig)

    def fit(self) -> None:
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        ema_accs = [] if self.ema is not None else None
        loglrs = []

        best_val_acc = 0
        best_ema_acc = 0

        for epoch in tqdm.tqdm(range(self.epochs), desc='Training Progress') if self.verbosity >= 1 else range(self.epochs):
            train_loss, train_acc = self._update_params(epoch=epoch)
            val_loss, val_acc = self._validate(epoch=epoch, model_to_validate='main')
            if ema_accs is not None:
                _, ema_acc = self._validate(epoch=epoch, model_to_validate='ema')
                ema_accs.append(ema_acc)
                if ema_acc > best_ema_acc:
                    best_ema_acc = ema_acc
                    torch.save(self.ema.state_dict(), os.path.join(self.save_directory, 'ema.pt')) #type: ignore
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            loglrs.append(math.log10(self.scheduler.get_last_lr()[-1])) #type: ignore

            torch.save(self.model.state_dict(), os.path.join(self.save_directory, 'latest.pt'))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.save_directory, 'best.pt'))

            self._plot(train_accs=train_accs, val_accs=val_accs, ema_accs=ema_accs, learning_rates=loglrs, epoch=epoch+1)

            self.scheduler.step()


def main():
    config = load_config()
    model_config = config.get('model')
    model = model_factory(**model_config).cuda() #type: ignore
    #torch.compile(model)
    '''trt_model = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
        enabled_precisions={torch.float16, torch.float8_e4m3fn},
    )'''
    model.calculate_flops()
    model.compile()

    train_config = config.get('config_train', dict())
    optimizer = get_optimizer(model=model, train_config=train_config)
    scheduler = get_scheduler(optimizer=optimizer, train_config=train_config)

    dataset_config = config.get('dataset', dict())
    train_dataloader = dataloader_factory(dataset_config=dataset_config, subset='train', transforms=get_transforms('train', dataset_config=dataset_config))
    val_dataloader = dataloader_factory(dataset_config=dataset_config, subset='val', transforms=get_transforms('val', dataset_config=dataset_config))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=nn.CrossEntropyLoss(),
        scheduler=scheduler,
        train_config=train_config
    )

    trainer.fit()

if __name__ == '__main__':
    main()