# Deep Learning Model Collection

A comprehensive PyTorch-based deep learning framework featuring implementations of popular computer vision architectures with advanced training capabilities, mixed precision support, and NVIDIA Transformer Engine integration.

## 🏗️ Architecture Overview

This repository provides a modular framework for training computer vision models with the following key features:

- **Multiple Model Architectures**: ResNet, EfficientNet, ConvNeXt, Vision Transformer, SimpleCNN variants, and MLPNet
- **Advanced Training Features**: Mixed precision (FP16/FP8), exponential moving averages (EMA), stochastic depth
- **NVIDIA Transformer Engine**: Optimized transformer layers with FP8 support
- **Flexible Configuration**: YAML-based configuration system
- **Factory Pattern**: Modular model, optimizer, and scheduler factories

## 🚀 Supported Models

### Convolutional Neural Networks
- **ResNet**: Standard, Bottleneck, and Pre-activation variants with stochastic depth
- **EfficientNet**: Mobile-optimized architecture with inverted residual blocks
- **ConvNeXt**: Modern ConvNet with transformer-inspired design
- **SimpleCNN**: Custom lightweight architectures (v1 and v2)

### Transformer Models
- **Vision Transformer (ViT)**: Pure transformer architecture for image classification
- **Experimental Modules**: Custom attention mechanisms and FFN variants

### Specialized Architectures
- **MLPNet**: Multi-layer perceptron variants (MLPNet-13, 25, 46, 136)

## 📋 Requirements

```bash
# Core dependencies
torch>=2.0.0
torchvision
transformer-engine-pytorch
einops
calflops
tqdm
matplotlib
opencv-python
pyyaml

# Optional for enhanced performance
torch-tensorrt  # TensorRT integration
```

## 🔧 Installation

```bash
# Clone the repository
git clone <repository-url>
cd deep-learning-models

# Install dependencies
pip install -r requirements.txt

# Set environment variables for Transformer Engine
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=1
```

## 📊 Usage

### Basic Training

1. **Configure your experiment** in a YAML file:

```yaml
model:
  model_type: "resnet"
  channel_dims: [64, 128, 256, 512]
  layer_numbers: [2, 2, 2, 2]
  block_types: ["standard", "standard", "standard", "standard"]
  n_classes: 1000
  p_stochastic_depth: 0.1
  stochastic_depth_linear_decay: true

dataset:
  n_classes: 1000
  batch_size: 128
  num_workers: 8
  crop_resolution: 224
  train:
    path: "/path/to/train/dataset"
    pre_augmentations:
      - type: "horizontal_flip"
        p: 0.5
      - type: "rotation"
        degrees: 10
  val:
    path: "/path/to/val/dataset"

config_train:
  epochs: 100
  precision: "fp16"  # Options: fp32, fp16, fp8
  save_directory: "./checkpoints"
  ema_beta: 0.9999
  ema_step: 32
  optimizer:
    optimizer_type: "adamw"
    lr: 0.001
    weight_decay: 0.05
  scheduler:
    - type: "linear"
      start_factor: 0.1
      total_iters: 10
      epoch: 10
    - type: "cosine_annealing"
      T_max: 90
```

2. **Run training**:

```bash
python train.py --config config.yaml
```

### Model Factory Usage

```python
from factory import model_factory, optimizer_factory, lr_scheduler_factory

# Create a model
model = model_factory(
    model_type="resnet",
    channel_dims=[64, 128, 256, 512],
    layer_numbers=[2, 2, 2, 2],
    block_types=["standard"] * 4,
    n_classes=1000
)

# Create optimizer
optimizer = optimizer_factory(
    optimizer_type="adamw",
    parameters=model.parameters(),
    lr=0.001,
    weight_decay=0.05
)

# Create scheduler
scheduler_configs = [
    {"type": "linear", "start_factor": 0.1, "total_iters": 10, "epoch": 10},
    {"type": "cosine_annealing", "T_max": 90}
]
scheduler = lr_scheduler_factory(optimizer, scheduler_configs)
```

### Individual Model Usage

```python
from model import ConvNeXt
from resnet import ResNet
from vision_transformer import VisionTransformer

# ConvNeXt
convnext = ConvNeXt(
    channel_dims=[96, 192, 384, 768],
    layer_numbers=[3, 3, 9, 3],
    p_stochastic_depth=0.1,
    stochastic_depth_linear_decay=True,
    n_classes=1000
)

# ResNet with stochastic depth
resnet = ResNet(
    channel_dims=[64, 128, 256, 512],
    layer_numbers=[2, 2, 2, 2],
    block_types=["standard"] * 4,
    n_classes=1000,
    p_stochastic_depth=0.1,
    stochastic_depth_linear_decay=True
)

# Vision Transformer
vit = VisionTransformer(
    in_channels=3,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    patch_size=16,
    num_classes=1000,
    ffn_multiplier=4,
    seq_len=196,
    batch_size=32
)
```

## 🔬 Advanced Features

### Mixed Precision Training
- **FP16**: Automatic mixed precision with GradScaler
- **FP8**: NVIDIA Transformer Engine FP8 support for maximum efficiency
- **Dynamic Loss Scaling**: Automatic gradient scaling management

### Exponential Moving Averages (EMA)
- Configurable EMA with custom decay rates
- Separate validation for EMA models
- Automatic best model checkpointing

### Stochastic Depth
- Per-layer stochastic depth with linear decay
- Configurable survival probabilities
- Improved training efficiency for deep networks

### Data Augmentation
- Pre-augmentation: Random rotation, horizontal flip, RandAugment
- Post-augmentation: CutMix, MixUp with configurable probabilities
- Automatic transform composition

## 📁 Project Structure

```
├── model.py                  # EfficientNet and ConvNeXt implementations
├── resnet.py                 # ResNet variants with stochastic depth
├── vision_transformer.py     # Vision Transformer implementation
├── simple.py                 # SimpleCNN architectures
├── mlpnet.py                 # MLP-based models
├── experimental_modules.py   # Custom modules and layers
├── factory.py                # Model, optimizer, and data factories
├── train.py                  # Main training script
├── config.yaml               # Example configuration file
└── requirements.txt          # Dependencies
```

## 🎯 Key Features

### Model Capabilities
- **Automatic FLOP Counting**: Built-in computational complexity analysis
- **Weight Initialization**: Proper weight initialization for all architectures
- **Gradient Checkpointing**: Memory-efficient training for large models
- **Model Compilation**: PyTorch 2.0 compile support

### Training Features
- **Flexible Scheduling**: Multi-stage learning rate scheduling
- **Automatic Checkpointing**: Best and latest model saving
- **Real-time Plotting**: Training metrics visualization
- **Comprehensive Logging**: Detailed training progress tracking

### Performance Optimizations
- **CUDA Optimizations**: cuDNN benchmark mode, optimized data loading
- **Memory Efficiency**: Gradient accumulation, mixed precision training
- **Multi-GPU Support**: Ready for distributed training setups

## 🔧 Configuration Options

### Dataset Configuration
```yaml
dataset:
  n_classes: 1000
  batch_size: 128
  num_workers: 8
  crop_resolution: 224
  train:
    path: "/path/to/train"
    pre_augmentations: [...]
    post_augmentations: [...]
  val:
    path: "/path/to/val"
```

### Training Configuration
```yaml
config_train:
  epochs: 100
  precision: "fp16"          # fp32, fp16, fp8
  save_directory: "./checkpoints"
  verbosity: 2               # 0: silent, 1: progress, 2: detailed
  ema_beta: 0.9999          # EMA decay rate
  ema_step: 32              # EMA update frequency
```

## 🚀 Performance Tips

1. **Use mixed precision** (`fp16` or `fp8`) for faster training
2. **Enable EMA** for better model generalization
3. **Tune batch size** based on your GPU memory
4. **Use stochastic depth** for deeper models
5. **Configure data augmentation** appropriate for your dataset

## 📈 Monitoring

The training script automatically generates:
- **Real-time plots**: Training/validation accuracy and learning rate curves
- **Checkpoints**: Best and latest model weights
- **Logs**: Detailed training metrics and progress

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🔗 References

- [ResNet](https://arxiv.org/abs/1512.03385)
- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [ConvNeXt](https://arxiv.org/abs/2201.03545)
- [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [Stochastic Depth](https://arxiv.org/abs/1603.09382)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)

---

For questions or issues, please open an issue in the repository.
