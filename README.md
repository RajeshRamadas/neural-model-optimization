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
├── src/
│   ├── __init__.py
│   ├── nas/                        # Neural Architecture Search
│   │   ├── proxyless/
│   │   ├── darts/
│   │   └── shared/
│   ├── detection/                  # Object Detection Models
│   │   ├── fast_rcnn/
│   │   ├── yolo/
│   │   ├── efficientnet/
│   │   └── shared/
│   ├── knowledge_distillation/     # Knowledge Distillation
│   │   ├── response_based/
│   │   ├── feature_based/
│   │   ├── multi_teacher/
│   │   └── shared/
│   ├── compression/                # Model Compression
│   │   ├── pruning/
│   │   ├── quantization/
│   │   └── shared/
│   └── utils/                      # Shared Utilities
├── configs/                        # Configuration Files
│   ├── nas/
│   ├── detection/
│   ├── knowledge_distillation/
│   └── compression/
├── examples/                       # Usage Examples
│   ├── nas/
│   ├── detection/
│   ├── knowledge_distillation/
│   └── compression/
└── tests/                          # Test Suite
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

## Framework Compatibility

The framework ensures compatibility between NNI and OFA through:

1. **Shared Search Spaces**: Common search space definitions across frameworks
2. **Compatible APIs**: Unified interfaces for key operations
3. **Conversion Utilities**: Tools to convert models between frameworks
4. **Integration Tests**: Ensuring cross-framework compatibility

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft's Neural Network Intelligence (NNI) team
- MIT's Once-for-All (OFA) research team

## Feature Compatibility: NNI + OFA

| **Feature** | **NNI (Microsoft)** | **Once-for-All (OFA, MIT)** | **Compatible?** |
|-------------|---------------------|----------------------------|----------------|
| **NAS** | ProxylessNAS, DARTS | ProxylessNAS, DARTS | Yes |
| **Object Detection Models** | Fast R-CNN, YOLO, EfficientNet | Fast R-CNN, YOLO, EfficientNet | Yes |
| **Knowledge Distillation (KD)** | Response, Feature-based, Multi-teacher | Response, Feature-based, Multi-teacher | Yes |
| **Model Compression** | Pruning & Quantization | Pruning & Quantization | Yes |

## Why Use NNI + OFA Together?

- **NNI excels at NAS & Hyperparameter tuning** → Finds the best student model
- **OFA is great for hardware-aware NAS & model compression** → Optimizes for edge deployment
- **Both support Pruning, Quantization, and KD** → Ensures compact & efficient models
- **Both support ProxylessNAS & DARTS** → Enables efficient NAS

## Workflow: Using NNI + OFA for NAS + KD + Pruning + Quantization

1. **Use NNI for NAS** → Select the best student model architecture
2. **Train the Student Model with KD** (using NNI or OFA's built-in KD)
3. **Apply OFA's Hardware-Aware NAS** → Optimize for CPU, GPU, Edge
4. **Pruning & Quantization** using **OFA & NNI** → Reduce model size & latency
