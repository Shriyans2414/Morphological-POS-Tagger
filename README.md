# Morphological Feature Integration for POS Tagging

## Abstract

Part-of-Speech (POS) tagging is a fundamental task in Natural Language Processing, heavily influenced by contextual and morphological properties of words. This project investigates how **morphological feature integration** can enhance POS tagging performance using a sequence modeling framework.

We implement a neural architecture that learns from **CoNLL-U formatted linguistic data** and jointly models syntactic and morphological information. Experiments are conducted on multilingual datasets to evaluate the effectiveness of morphological signals in improving tagging accuracy.

---

## Problem Statement

Traditional POS tagging models rely primarily on word-level context, often overlooking rich **morphological attributes** such as tense, gender, number, and case.

This project addresses:

* How can morphological features improve POS tagging?
* Can a unified model learn both syntactic and morphological patterns effectively?
* How does this impact performance across different languages?

---

## Methodology

### 1. Data Processing

* Custom parser (`parse_conllu.py`) processes `.conllu` files
* Extracts:

  * Tokens
  * POS tags
  * Morphological features

### 2. Feature Representation

* Tokens are encoded into sequences
* Morphological features are structured and integrated into the learning pipeline

### 3. Sequence Modeling

* A **Bidirectional LSTM-based architecture** is used to:

  * Capture forward and backward context
  * Learn dependencies across sequences

### 4. Tasks

* POS Tag Prediction (`bilstm_pos.py`)
* Morphological Feature Modeling (`bilstm_morph.py`)

---

## System Pipeline

```
CoNLL-U Dataset
     ↓
Parsing (Tokens + POS + Morph Features)
     ↓
Sequence Encoding
     ↓
Bi-directional Context Learning
     ↓
POS Tag Prediction
     ↓
Evaluation & Result Logging
```

---

## Dataset

This project uses datasets from Universal Dependencies.

### Languages:

* English — EWT (en_ewt)
* Hindi — HDTB (hi_hdtb)

### Format:

* `.conllu` (Universal Dependencies standard)

---

## Implementation Details

### Core Files:

* `parse_conllu.py` → Dataset parsing and preprocessing
* `bilstm_pos.py` → POS tagging model
* `bilstm_morph.py` → Morphological feature model

### Libraries Used:

* Python
* NumPy
* Pandas
* Deep Learning Framework (PyTorch / TensorFlow)

---

## Project Structure

```
Morphological-POS-Tagger/
│── data/              # UD datasets
│── src/
│   ├── bilstm_pos.py
│   ├── bilstm_morph.py
│   ├── parse_conllu.py
│── results/
│   └── experiment_results.csv
│── requirements.txt
│── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/Morphological-POS-Tagger.git
cd Morphological-POS-Tagger
pip install -r requirements.txt
```

---

## Execution

Run POS tagging model:

```bash
python src/bilstm_pos.py
```

Run morphological feature model:

```bash
python src/bilstm_morph.py
```

---

## Results

Results are recorded in:

```
results/experiment_results.csv
```

### Evaluation Includes:

* POS tagging performance
* Morphological prediction capability
* Cross-lingual observations
* Accuracy of CRF and BiLSTM Models

---

## Key Observations

* Morphological features improve contextual understanding
* Performance gains are more prominent in morphologically rich languages (e.g., Hindi)
* Joint learning enhances representation quality

---

## Future Work

* Transformer-based architectures (BERT, RoBERTa)
* Attention-based feature fusion
* Low-resource language adaptation
* Real-time deployment via APIs

---

## Contributions

* Integration of morphological features into POS tagging pipeline
* Custom CoNLL-U parsing pipeline
* Multilingual evaluation (English & Hindi)
* Practical implementation of sequence labeling models

---

## License

MIT License

---

## Acknowledgment

This work utilizes datasets from the Universal Dependencies project, a widely used benchmark for syntactic and morphological analysis.
