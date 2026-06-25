# Neural Network to Describe Simple Pictures — Group 40

Coursework project comparing three image-to-text architectures on three synthetic datasets.
Core question: how does caption representation (sequence vs structured vs embedding) affect what a model can learn?

**Headline result:** MultiHead CNN achieves **96.67% sentence accuracy on TicTacToe**, outperforming GPT-3.5 (82%) on the same task.

---

## My Contribution — Ishaan Bhalla

- **Led all model development** — designed and implemented all three neural network architectures (Embedding CNN, CNN+LSTM, MultiHead CNN) from scratch
- **Implemented `get_metadata_files`** — adaptive dataset loader that detects whether data is stored as a single JSONL file (Shapes, Numbers) or three split files (TicTacToe) and standardises loading across both formats
- **Built the inference pipeline and demo app** — `src/inference/` modules and the Streamlit app (`app.py`) for interactive caption generation
- Set up repository structure, coding standards, and the artifacts output pipeline (`/artifacts`)

---

## Repository Structure

```
├── src/
│   ├── data/        — dataset generation scripts and PyTorch Dataset classes
│   ├── models/      — model definitions (multihead_cnn, cnn_lstm, embedding_cnn)
│   ├── training/    — training scripts
│   └── inference/   — inference modules for the demo app
├── artifacts/       — saved model weights and metrics per model/dataset
├── data/processed/  — generated images and JSONL metadata
├── Evaluation/      — evaluation script and result JSON files
├── app.py           — Streamlit demo app
├── docs/            — project documentation
└── tests/           — dataset smoke tests
```

---
## Demo App

Upload a synthetic image or sample from the test set and get a predicted caption from either model.


---

## Datasets

All images are synthetically generated using PIL.

| Dataset | Images | Size | Caption format |
|---|---|---|---|
| **Shapes** | 1,000 | 256×256 | `a [size] [color] [shape] is [relation] a [size] [color] [shape]` |
| **Numbers** | 1,000 | 128×128 | `a [size] [color] [number]` |
| **TicTacToe** | 300 | 256×256 | `X is in [positions]; O is in [positions]` |

Splits: 80% train / 10% val / 10% test across all datasets.

---

## Architectures

### Embedding CNN (retrieval baseline)
Image → CNN → 384-dim vector → cosine similarity → nearest caption from pool
Text embeddings from `sentence-transformers/all-MiniLM-L6-v2`.

### CNN + LSTM (sequence generation)
Image → CNN encoder → 256-dim feature → LSTM decoder → token sequence
Trained with teacher forcing; greedy decode at inference.

### MultiHead CNN (structured prediction)
Image → 4-block CNN backbone → 512-dim feature → N independent classification heads → caption reconstructed from template
One head per attribute (e.g. shape, colour, relation). Loss = sum of per-head cross-entropies.
Key design: `AdaptiveAvgPool2d(4×4)` instead of global pooling to preserve spatial structure for relation prediction.

---

## Results

### Model comparison (test set sentence accuracy)

| Task | CNN-LSTM | MultiHead CNN |
|---|---|---|
| **TicTacToe** | — | **96.67%** |
| **Shapes** | 0% | 25.00% |
| **Numbers** | — | 4.00% |

CNN-LSTM token accuracy: 74.49% (TicTacToe), 61.53% (Shapes), 56.00% (Numbers).
CNN-LSTM sentence accuracy is near zero on Shapes and Numbers — the combinatorial caption space is too large without pretraining.

### MultiHead CNN vs GPT-3.5

| Task | Metric | MultiHead CNN | GPT-3.5 |
|---|---|---|---|
| TicTacToe | Token accuracy | **99.26%** | 91% |
| TicTacToe | Sentence accuracy | **96.67%** | 82% |
| Shapes | Token accuracy | 65.71% | 74% |
| Shapes | Sentence accuracy | 25.00% | 48% |
| Numbers | Token accuracy | 69.86% | 93% |
| Numbers | Sentence accuracy | 4.00% | 72% |

The MultiHead CNN surpasses GPT-3.5 on TicTacToe because the task maps cleanly to independent per-cell classification — exactly what the multi-head structure is designed for.

### Per-attribute accuracy (MultiHead CNN, test set)

**TicTacToe** — all 9 board positions ≥ 96.7% (most at 100%)

**Shapes**
| Attribute | Accuracy |
|---|---|
| Object 1 size | 79% |
| Object 1 color | 65% |
| Object 1 shape | 64% |
| Relation | 50% |
| Object 2 size | 83% |
| Object 2 color | 59% |
| Object 2 shape | 60% |

Relation prediction is the bottleneck at 50% (chance for 4 classes).

**Numbers**
| Attribute | Accuracy |
|---|---|
| Color | 82% |
| Length | 97% |
| Digit 3 | 88% |
| Digit 0 | 60% |
| Size | 47% |

---

## Setup

**Requirements:** Python 3.14, PyTorch, Streamlit

```bash
git clone https://github.com/ishaan-bhalla/Neural-Network-to-Describe-Simple-Pictures-Group-40.git
cd Neural-Network-to-Describe-Simple-Pictures-Group-40
pip install -r requirements.txt
```

### Generate datasets

```bash
python src/data/generate_shapes.py
python src/data/generate_numbers.py --num_samples 1000
python src/data/generate_tictactoe.py --num-samples 300
```

### Train models

```bash
# MultiHead CNN
python -m src.training.train_multihead_cnn --dataset shapes
python -m src.training.train_multihead_cnn --dataset numbers
python -m src.training.train_multihead_cnn --dataset tictactoe

# CNN + LSTM
python -m src.training.train_cnn_lstm --dataset shapes
python -m src.training.train_cnn_lstm --dataset tictactoe
```

Trained weights are saved to `artifacts/{dataset}/{model}/best_model.pth`.

### Run the demo

```bash
streamlit run app.py
```

## Resume Bullet

> **Image Captioning with Neural Networks** | Python, PyTorch, CNN, LSTM, Sentence-BERT
> Group project comparing three image-to-text architectures (Embedding CNN, CNN+LSTM, MultiHead CNN) on synthetic datasets. Led all model development; implemented adaptive dataset loading to handle inconsistent formats across three dataset types. MultiHead CNN achieved **96.67% sentence accuracy on tic-tac-toe, outperforming GPT-3.5** (82%). Evaluated using token accuracy, sentence accuracy, and per-attribute head accuracy metrics.
