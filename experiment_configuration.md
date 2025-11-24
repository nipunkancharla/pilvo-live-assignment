# PII Entity Recognition Experiment Details

## 1. Model Architecture

- **Base Model:** `prajjwal1/bert-mini`
- **Tokenizer:** `AutoTokenizer` (from `prajjwal1/bert-mini`)
- **Reason for Choice:** \* Initial experiments with `distilbert-base-uncased` yielded ~80ms CPU latency, violating the 20ms constraint.
  - `bert-mini` (4 layers, 256 hidden) provided the optimal trade-off, achieving ~12ms latency with strong PII precision (>0.90).

## 2. Training Hyperparameters

- **Epochs:** 10
- **Batch Size:** 16
- **Learning Rate:** 5e-4
- **Optimizer:** AdamW
- **Max Sequence Length:** 256
- **Loss Function:** CrossEntropyLoss (Standard Token Classification)

## 3. Data Strategy (Synthetic Generation)

- **Generation Tool:** Python `Faker` library + Custom Noise Script.
- **Dataset Size:** \* Train: 1000 samples
  - Dev: 100 samples
- **Noise Simulation:** \* Randomly converted digits to words (e.g., "4" -> "four").
  - Replaced punctuation with spoken equivalents ("." -> "dot", "@" -> "at").
  - Removed all standard punctuation and casing to simulate raw STT output.

## 4. Inference & Latency Optimization

- **Target:** < 20 ms p95 latency on CPU.
- **Optimization Implemented:** PyTorch Dynamic Quantization (`qint8`).
  - Applied to `torch.nn.Linear` layers during inference.
  - Reduced model size and computational cost without retraining.
- **Final Latency (p95):** ~12ms (Batch Size: 1, CPU).

## 5. Performance Metrics (Dev Set)

- **Macro F1:** 1.00
- **PII Precision:** 1.00 (Exceeds 0.80 target)
- **Recall:** 1.00

## 6. Colab link

- for more details you can lookup this collabnotebook I used for training
- https://colab.research.google.com/drive/1EK_pz7ID7lP_95n6GRQo9ggtQP1DLSYI?usp=sharing
