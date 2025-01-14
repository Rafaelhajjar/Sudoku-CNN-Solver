# Sudoku CNN Solver: Deep Learning for Puzzle Solving

A sophisticated deep learning approach to solving Sudoku puzzles using Convolutional Neural Networks (CNNs). This project demonstrates how modern deep learning techniques can be applied to solve complex constraint satisfaction problems traditionally handled by rule-based algorithms.

## Table of Contents
- [Project Overview](#project-overview)
- [Technical Architecture](#technical-architecture)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Implementation Approaches](#implementation-approaches)
- [Future Improvements](#future-improvements)

## Project Overview <a name="project-overview"></a>

This solver uses a deep CNN architecture to learn the spatial patterns and constraints inherent in Sudoku puzzles. Rather than explicitly programming Sudoku rules, the network learns to understand valid number placements through training on millions of puzzle-solution pairs.

### Key Features
- Pure deep learning approach without rule-based post-processing
- 10-block deep CNN architecture with batch normalization
- Iterative solving process using confidence-based predictions
- Supports puzzles of varying difficulty levels
- Achieves high accuracy on standard 9x9 Sudoku grids

## Technical Architecture <a name="technical-architecture"></a>

### Neural Network Design
- **Input Layer**: 9x9 grid (81 cells) representing the puzzle
- **Hidden Layers**: 10 blocks of convolutional layers
  - Filter Size: 3x3
  - Number of Filters: 512 per layer
  - Activation: ReLU
  - Normalization: Batch Normalization
- **Output Layer**: 9x9x10 (probability distribution for each cell)

### Solving Process
1. Initial grid analysis through CNN
2. Confidence-based cell filling
3. Iterative prediction refinement
4. Final solution validation

## Requirements <a name="requirements"></a>

### Core Dependencies
- NumPy >= 1.11.1
- TensorFlow == 1.1

### Hardware Requirements
- Recommended: GPU with CUDA support
- Minimum 8GB RAM for training
- 2GB RAM for inference

## Dataset <a name="dataset"></a>

The training dataset consists of 1M Sudoku puzzles generated using our custom generator. The dataset includes:
- 1 million training puzzles
- Various difficulty levels
- Puzzle-solution pairs for supervised learning

[Download Dataset](https://www.kaggle.com/bryanpark/sudoku/downloads/sudoku.zip)

## Model Performance <a name="model-performance"></a>

### Experimental Results

#### Epoch Performance
| Epochs | Accuracy (%) |
|--------|-------------|
| 3      | 74          |
| 5      | 76          |
| 10     | 78          |

#### Architecture Variations
| Number of Blocks | Accuracy (%) |
|-----------------|-------------|
| 10              | 72          |
| 12              | 74          |
| 15              | 81          |
| 18              | 78          |
| 20              | 74          |

#### Filter Configuration
| Number of Filters | Accuracy (%) |
|------------------|-------------|
| 3                | 72          |
| 6                | 71          |
| 9                | OOM*        |

*OOM: Out of Memory

#### Regularization Impact
| Dropout Rate | Accuracy (%) |
|-------------|-------------|
| 0.2         | 66          |
| 0.05        | 77          |

#### L2 Regularization
| L2 Value | Accuracy (%) |
|----------|-------------|
| 0.0001   | 74          |
| 0.001    | 77          |
| 0.002    | 73          |
| 0.01     | 65          |

## Project Structure <a name="project-structure"></a>

- `data_load.py` - Data loading and preprocessing utilities
- `generate_sudoku.py` - Puzzle generation module
- `hyperparams.py` - Model hyperparameters configuration
- `modules.py` - Neural network components
- `train.py` - Training implementation
- `test.py` - Testing and evaluation module

## Implementation Approaches <a name="implementation-approaches"></a>

### CNN Implementation
Our primary implementation uses a deep CNN architecture with:
- 10 convolutional blocks
- Batch normalization layers
- ReLU activation functions
- Confidence-based prediction system

### RNN Approach (Experimental)
The RNN implementation uses bidirectional LSTM cells with:
- Cell-hint tuple system for move ordering
- Probability-based solution selection
- Iterative board updating
- Joint probability calculations

### Deep Reinforcement Learning (Proposed)
Future implementation plans include:
- State-action space modeling
- Reward function design
- Policy gradient methods
- Value function approximation

## Future Improvements <a name="future-improvements"></a>

### Architecture Optimization
- Experiment with residual connections
- Test different normalization techniques
- Optimize filter configurations

### Training Improvements
- Implement curriculum learning
- Add data augmentation
- Explore transfer learning

### Solving Strategies
- Implement beam search
- Add probabilistic backtracking
- Develop hybrid solving approaches

### Research Directions
- Investigate attention mechanisms
- Explore transformer architectures
- Test reinforcement learning approaches



---
**Note**: This is an active research project. Performance metrics and features are continuously being updated.





