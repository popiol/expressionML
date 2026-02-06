# expressionML

A machine learning framework for teaching neural networks to learn and perform mathematical expressions and operations through knowledge representation and agent-based training.

## Overview

expressionML is an experimental Python framework that explores how neural networks can learn mathematical operations (arithmetic, bitwise operations, etc.) by representing knowledge as embeddings and training agents to perform transformations on that knowledge.

## Features

- **Multiple ML Model Architectures**: 8 different model versions (v0-v7) with varying complexity
  - Dense fully-connected networks
  - Convolutional architectures
  - Stateful networks with residual connections
  - Custom layer implementations (TrainableMask, PadLastDimLayer)

- **Flexible Knowledge Representation**: 
  - Custom encoding/decoding system for numerical values
  - Embedding-based knowledge format
  - Support for both float and integer coders

- **Multiple Training Tasks**:
  - Arithmetic operations (addition, subtraction, multiplication, division)
  - Bitwise operations (AND, OR, XOR)
  - Cyclic bit shifts
  - One-hot encoding
  - Value selection (chooser)
  - Identity mapping

- **Agent-Based Learning**:
  - Supervised and reinforcement learning modes
  - Exploration mode with noise injection
  - Feedback-based training with priority queue
  - Dynamic model creation based on input/output formats


## Architecture

### Core Components

- **Agent** (`src/agent.py`): Manages the learning process, model selection, and training feedback
- **MlModel** (`src/ml_model.py`): Neural network wrapper with multiple architecture variants
- **Dataset** (`src/dataset.py`): Generates training data for various mathematical operations
- **Knowledge** (`src/knowledge.py`): Handles knowledge representation and encoding/decoding
- **Coder** (`src/coder.py`): Encodes numerical values into fixed-size embeddings
- **Runner** (`src/runner.py`): Orchestrates training and evaluation loops
