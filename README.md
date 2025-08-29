# ArXiv Paper Category Classification

A machine learning project that classifies ArXiv research papers into their respective computer science categories using a fine-tuned large language model.

## Overview

This project fine-tunes a LLaMA model to classify academic papers from ArXiv into the following categories:
- Machine Learning
- Computer Vision and Pattern Recognition 
- Computation and Language (Natural Language Processing)
- Robotics
- Cryptography and Security
- Artificial Intelligence

The model takes a paper's title and abstract as input and predicts the most appropriate category.

## Project Structure

```
├── data/
│   └── arxiv_dataset.csv       # Dataset containing ArXiv papers
├── data_utils.py              # Data processing utilities
├── evaluation.py             # Model evaluation code
├── finetune.py              # Model fine-tuning script  
├── inference.py             # Inference script for predictions
└── requirements.txt         # Python dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the following dependencies installed:
- transformers==4.45.2
- numpy==2.1.2  
- torch==2.4.1
- pillow==10.4.0
- pandas==2.2.3
- datasets==3.0.1
- peft==0.13.1
- tabulate==0.9.0
- accelerate==1.0.0

## Usage

### Training

To fine-tune the model:

```bash
python finetune.py
```

The fine-tuning script:
- Uses LoRA (Low-Rank Adaptation) for efficient training
- Implements gradient accumulation and mixed precision
- Saves checkpoints of the best model based on validation accuracy

### Inference

To run inference on a paper:

```bash 
python inference.py "2410.08196"
```

Replace the ArXiv ID with any valid paper ID to get its predicted category.

### Evaluation

To evaluate model performance:

```bash
python evaluation.py
```

This will run the model on a test set and output metrics including:
- Classification accuracy
- Invalid prediction rate
- Confusion matrix

## Model Details

- Base model: meta-llama/Llama-3.2-1B-Instruct
- Fine-tuning method: LoRA with rank 64
- Training data: ArXiv papers with title and abstract
- Output: One of 6 predefined computer science categories

## Results

The model achieves competitive performance in classifying CS papers into their respective categories, with key metrics displayed during evaluation.

## License

This project uses the meta-llama/Llama model which requires a license from Meta. Please ensure you have appropriate permissions before using the model.

## Acknowledgments

The ArXiv dataset used in this project is derived from papers submitted to arXiv.org.
