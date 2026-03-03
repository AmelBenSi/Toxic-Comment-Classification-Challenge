# Toxic-Comment-Classification-Challenge
Multi-label toxic comment classification on Wikipedia talk-page comments (Kaggle). Predicts probabilities for 6 toxicity types (toxic, severe_toxic, obscene, threat, insult, identity_hate) using an end-to-end NLP notebook imported from Google Colab.

---

## What this repo does
It builds a **multi-label** text classifier for toxic comments using:
- **Data preprocessing**
- **Feature extraction (vectorization)** — testing two methods:
  - **TF-IDF vectors**
  - **Word embeddings**
- Multiple ML models wrapped for multi-label prediction
- Evaluation with **Accuracy** + **ROC-AUC** per label

---

## The journey (pipeline)
1. **Load** the dataset  
2. **Explore** labels + class distribution  
3. **Process** text (cleaning + normalization)  
4. **EDA** (tokens/sentences per class, common words per label)  
5. **Feature Extraction**
   - **TF-IDF**
   - **spaCy vectors** (`doc.vector`)
6. **Model Training** (multi-output)
7. **Predict** & prepare a submission file

---

## Models included
The notebook experiments with classic ML baselines (multi-label setup via `MultiOutputClassifier`):
- KNN
- SGDClassifier
- Logistic Regression
- Random Forest
- MLPClassifier

---

## What you need
- Python 3.9+ recommended
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `spacy`
- spaCy model: **`en_core_web_lg`** (required for embeddings)

Install the spaCy model:
```bash
python -m spacy download en_core_web_lg
