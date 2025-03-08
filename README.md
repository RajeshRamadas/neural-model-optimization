# Neural Model Optimization Framework

A comprehensive framework for neural network optimization that integrates Microsoft's Neural Network Intelligence (NNI) and MIT's Once-for-All (OFA) frameworks, supporting neural architecture search, object detection models, knowledge distillation, and model compression techniques.

## Overview

This repository provides a unified approach to neural network optimization by leveraging the strengths of both NNI and OFA frameworks. It supports various techniques including:

- **Neural Architecture Search (NAS)**: ProxylessNAS, DARTS
- **Object Detection Models**: Fast R-CNN, YOLO, EfficientNet
- **Knowledge Distillation (KD)**: Response-based, Feature-based, Multi-teacher
- **Model Compression**: Pruning & Quantization

The framework is designed to be modular, allowing users to leverage either NNI or OFA implementations while maintaining compatibility between them.

## Repository Structure

```
neural-model-optimization/
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── docs/
│   ├── getting_started.md
│   ├── nni_integration.md
│   ├── ofa_integration.md
│   └── examples/
├── data/
│   ├── __init__.py
│   ├── datasets/                   # Raw datasets storage
│   │   ├── cifar10/
│   │   ├── cifar100/
│   │   ├── imagenet/
│   │   ├── coco/
│   │   ├── voc/
│   │   └── custom/
│   ├── processed/                  # Preprocessed datasets
│   │   ├── cifar10/
│   │   ├── cifar100/
│   │   ├── imagenet/
│   │   ├── coco/
│   │   ├── voc/
│   │   └── custom/
│   ├── downloaders/                # Dataset download scripts
│   │   ├── __init__.py
│   │   ├── cifar_downloader.py
│   │   ├── imagenet_downloader.py
│   │   ├── coco_downloader.py
│   │   └── voc_downloader.py
│   ├── loaders/                    # Dataset loading utilities
│   │   ├── __init__.py
│   │   ├── cifar_loader.py
│   │   ├── imagenet_loader.py
│   │   ├── coco_loader.py
│   │   └── voc_loader.py
│   ├── transforms/                 # Data transformations & augmentations
│   │   ├── __init__.py
│   │   ├── common_transforms.py
│   │   ├── classification_transforms.py
│   │   └── detection_transforms.py
│   └── utils/                      # Utility functions
│       ├── __init__.py
│       ├── data_checking.py
│       ├── data_cleaning.py
│       └── data_visualization.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # Base dataset classes
│   │   ├── dataloader.py          # Custom dataloader implementations
│   │   ├── samplers.py            # Custom sampling strategies
│   │   └── data_config.py         # Dataset configurations
│   ├── nas/                        # Neural Architecture Search
│   │   ├── __init__.py
│   │   ├── proxyless/
│   │   │   ├── __init__.py
│   │   │   ├── nni_integration.py
│   │   │   └── ofa_integration.py
│   │   ├── darts/
│   │   │   ├── __init__.py
│   │   │   ├── nni_integration.py
│   │   │   └── ofa_integration.py
│   │   └── shared/
│   │       ├── __init__.py
│   │       └── search_space.py
│   ├── detection/                  # Object Detection Models
│   │   ├── __init__.py
│   │   ├── fast_rcnn/
│   │   │   ├── __init__.py
│   │   │   ├── nni_integration.py
│   │   │   └── ofa_integration.py
│   │   ├── yolo/
│   │   │   ├── __init__.py
│   │   │   ├── nni_integration.py
│   │   │   └── ofa_integration.py
│   │   ├── efficientnet/
│   │   │   ├── __init__.py
│   │   │   ├── nni_integration.py
│   │   │   └── ofa_integration.py
│   │   └── shared/
│   │       ├── __init__.py
│   │       └── detection_utils.py
│   ├── knowledge_distillation/     # Knowledge Distillation
│   │   ├── __init__.py
│   │   ├── response_based/
│   │   │   ├── __init__.py
│   │   │   ├── nni_integration.py
│   │   │   └── ofa_integration.py
│   │   ├── feature_based/
│   │   │   ├── __init__.py
│   │   │   ├── nni_integration.py
│   │   │   └── ofa_integration.py
│   │   ├── multi_teacher/
│   │   │   ├── __init__.py
│   │   │   ├── nni_integration.py
│   │   │   └── ofa_integration.py
│   │   └── shared/
│   │       ├── __init__.py
│   │       └── distillation_utils.py
│   ├── compression/                # Model Compression
│   │   ├── __init__.py
│   │   ├── pruning/
│   │   │   ├── __init__.py
│   │   │   ├── nni_integration.py
│   │   │   └── ofa_integration.py
│   │   ├── quantization/
│   │   │   ├── __init__.py
│   │   │   ├── nni_integration.py
│   │   │   └── ofa_integration.py
│   │   └── shared/
│   │       ├── __init__.py
│   │       └── compression_utils.py
│   └── utils/                      # Shared Utilities
│       ├── __init__.py
│       ├── compatibility.py
│       ├── metrics.py
│       └── visualization.py
├── configs/                        # Configuration Files
│   ├── nas/
│   │   ├── proxyless_nni.json
│   │   ├── proxyless_ofa.json
│   │   ├── darts_nni.json
│   │   └── darts_ofa.json
│   ├── detection/
│   │   ├── fast_rcnn_nni.json
│   │   ├── fast_rcnn_ofa.json
│   │   ├── yolo_nni.json
│   │   ├── yolo_ofa.json
│   │   ├── efficientnet_nni.json
│   │   └── efficientnet_ofa.json
│   ├── knowledge_distillation/
│   │   ├── response_nni.json
│   │   ├── response_ofa.json
│   │   ├── feature_nni.json
│   │   ├── feature_ofa.json
│   │   ├── multi_teacher_nni.json
│   │   └── multi_teacher_ofa.json
│   └── compression/
│       ├── pruning_nni.json
│       ├── pruning_ofa.json
│       ├── quantization_nni.json
│       └── quantization_ofa.json
├── examples/                       # Usage Examples
│   ├── nas/
│   │   ├── proxyless_example.py
│   │   └── darts_example.py
│   ├── detection/
│   │   ├── fast_rcnn_example.py
│   │   ├── yolo_example.py
│   │   └── efficientnet_example.py
│   ├── knowledge_distillation/
│   │   ├── response_based_example.py
│   │   ├── feature_based_example.py
│   │   └── multi_teacher_example.py
│   └── compression/
│       ├── pruning_example.py
│       └── quantization_example.py
└── tests/                          # Test Suite
    ├── test_nas.py
    ├── test_detection.py
    ├── test_knowledge_distillation.py
    ├── test_compression.py
    └── test_compatibility.py
```

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neural-model-optimization.git
   cd neural-model-optimization
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Data Management

### Downloading Datasets

The framework provides utilities to download and prepare commonly used datasets:

```python
# Download CIFAR-10 dataset
from data.downloaders.cifar_downloader import download_cifar10
dataset_path = download_cifar10()

# Verify dataset integrity
from data.utils.data_checking import verify_image_dataset
stats, corrupt_files = verify_image_dataset(dataset_path)
print(f"Dataset stats: {stats}")
```

### Dataset Configuration

A unified configuration approach for using datasets with both NNI and OFA:

```python
from src.data.data_config import DataConfig

# Create a unified dataset configuration
cifar_config = DataConfig(
    name="cifar10",
    train_path="./data/datasets/cifar10/train",
    val_path="./data/datasets/cifar10/val",
    test_path="./data/datasets/cifar10/test",
    batch_size=128,
    input_size=(3, 32, 32),
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616)
)

# Convert to NNI format
nni_config = cifar_config.get_nni_dataset_config()

# Convert to OFA format
ofa_config = cifar_config.get_ofa_dataset_config()
```

## Usage Workflow

### 1. Select Optimization Technique

Choose the appropriate technique based on your requirements:

- **Neural Architecture Search**: For finding optimal network architectures
- **Object Detection Models**: For optimizing detection models
- **Knowledge Distillation**: For transferring knowledge from teacher to student models
- **Model Compression**: For reducing model size while maintaining performance

### 2. Choose Framework Integration

Decide whether to use NNI or OFA implementation:

- **NNI**: Provides extensive search space and trial management
- **OFA**: Offers efficient once-for-all network training approach

### 3. Configure Parameters

Modify the appropriate configuration file in the `configs/` directory or create a new one for your specific use case.

### 4. Run Optimization

Execute the appropriate example script or create your own using the provided modules:

```python
from src.nas.proxyless import nni_integration as proxyless_nni
from src.utils.metrics import evaluate_model

# Configure your optimization
config = {...}

# Run optimization
optimized_model = proxyless_nni.search(config)

# Evaluate results
metrics = evaluate_model(optimized_model, test_data)
print(metrics)
```

### 5. Evaluate and Deploy

Evaluate the optimized model and deploy it to your target environment.

## Examples

### Neural Architecture Search

```python
# Example of running ProxylessNAS with NNI
from src.nas.proxyless import nni_integration
nni_integration.run_search(dataset='cifar10', epochs=50)
```

### Knowledge Distillation

```python
# Example of feature-based knowledge distillation with OFA
from src.knowledge_distillation.feature_based import ofa_integration
ofa_integration.distill(teacher_model, student_model, dataset='imagenet')
```

### Model Compression

```python
# Example of model pruning with NNI
from src.compression.pruning import nni_integration
nni_integration.prune_model(model, config_path='configs/compression/pruning_nni.json')
```

## End-to-End Example: Classification Model Optimization

Here's a complete example that demonstrates the entire workflow:

```python
# 1. Dataset preparation
from data.downloaders.cifar_downloader import download_cifar10
from src.data.data_config import DataConfig

# Download and configure dataset
dataset_path = download_cifar10()
data_config = DataConfig(
    name="cifar10",
    train_path=f"{dataset_path}/train",
    val_path=f"{dataset_path}/val",
    test_path=f"{dataset_path}/test",
    batch_size=128
)

# 2. Neural Architecture Search with NNI
from src.nas.proxyless import nni_integration

# Find optimal architecture
search_config = {
    "data_config": data_config.get_nni_dataset_config(),
    "search_space": "proxyless",
    "max_epochs": 50,
    "max_trials": 20
}
best_architecture = nni_integration.run_search(search_config)

# 3. Knowledge Distillation
from src.knowledge_distillation.feature_based import ofa_integration

# Train teacher model
teacher_model = nni_integration.train_model(best_architecture)

# Train student model with KD
student_config = {
    "teacher_model": teacher_model,
    "data_config": data_config.get_ofa_dataset_config(),
    "distillation_alpha": 0.5,  # Weight of distillation loss
    "temperature": 4.0          # Temperature for soft targets
}
student_model = ofa_integration.distill(student_config)

# 4. Model Compression
from src.compression.pruning import nni_integration as pruning_nni
from src.compression.quantization import ofa_integration as quant_ofa

# Apply pruning
pruned_model = pruning_nni.prune_model(student_model, {
    "sparsity": 0.7,            # Target sparsity
    "pruning_method": "l1"      # L1-norm based pruning
})

# Apply quantization
quantized_model = quant_ofa.quantize_model(pruned_model, {
    "bits": 8,                  # 8-bit quantization
    "symmetric": True           # Use symmetric quantization
})

# 5. Evaluation
from src.utils.metrics import evaluate_model

# Evaluate final model
metrics = evaluate_model(quantized_model, data_config)
print(f"Final model metrics: {metrics}")
```

## Framework Compatibility

### Feature Compatibility: NNI + OFA

| **Feature** | **NNI (Microsoft)** | **Once-for-All (OFA, MIT)** | **Compatible?** |
|-------------|---------------------|----------------------------|----------------|
| **NAS** | ProxylessNAS, DARTS | ProxylessNAS, DARTS | Yes |
| **Object Detection Models** | Fast R-CNN, YOLO, EfficientNet | Fast R-CNN, YOLO, EfficientNet | Yes |
| **Knowledge Distillation (KD)** | Response, Feature-based, Multi-teacher | Response, Feature-based, Multi-teacher | Yes |
| **Model Compression** | Pruning & Quantization | Pruning & Quantization | Yes |

### Why Use NNI + OFA Together?

- **NNI excels at NAS & Hyperparameter tuning** → Finds the best student model
- **OFA is great for hardware-aware NAS & model compression** → Optimizes for edge deployment
- **Both support Pruning, Quantization, and KD** → Ensures compact & efficient models
- **Both support ProxylessNAS & DARTS** → Enables efficient NAS

### Workflow: Using NNI + OFA for NAS + KD + Pruning + Quantization

1. **Use NNI for NAS** → Select the best student model architecture
2. **Train the Student Model with KD** (using NNI or OFA's built-in KD)
3. **Apply OFA's Hardware-Aware NAS** → Optimize for CPU, GPU, Edge
4. **Pruning & Quantization** using **OFA & NNI** → Reduce model size & latency

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft's Neural Network Intelligence (NNI) team
- MIT's Once-for-All (OFA) research team