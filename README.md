# Review Rating Prediction 

- This project is focused on **predicting review sentiment** as a 3-class problem using a combination of **text features (TF-IDF)** and **custom-engineered metadata features**, powered by **XGBoost**.
- STREAMLIT URL : https://review-sentiment-analysis-by-saket.streamlit.app/
---

## 🧠 Objective And Demo

Classify customer reviews into:

- **-1** → Negative   
- **0** → Neutral   
- **1** → Positive 

<img width="1139" height="606" alt="example" src="https://github.com/user-attachments/assets/ee7cf1ce-5284-4643-b28f-aefbf5bc05c8" />

---

## ⚙️ Model & Features

### ✅ Text Features
- TF-IDF vectorization (`max_features=2000`)
- Preprocessing includes:
  - Lowercasing
  - Stopword filtering (custom meaningful stopwords retained)
  - Lemmatization
  - Handling emojis, hashtags, mentions, and punctuations like `!`, `?`

### ✅ Engineered Metadata Features
- `mojibake_freq` — count of mojibake and special characters
- `has_mojibake` — binary indicator
- `review_char`, `review_words`, `review_sent` — text statistics
- `repetition_score` — proportion of repeated words
- `max_word_repeat_ratio` — max single word frequency ratio
- `sentiment_score`, POS-tag counts

### ✅ Model
- `XGBClassifier` from the `xgboost` library
- Achieved **>85% accuracy** on test data 

---

## 📈 Performance

- ✅ Accuracy: **~85%**
---

## 💾 Saving & Loading

Model is saved using:
```python
import joblib
joblib.dump(xgb,'model.joblib')
joblib.dump(tfidf_review,'tfidf.joblib')
```
---
## 📦 DEPENDENCIES

Python 3.10+
`xgboost`
`scikit-learn`
`nltk`
`optuna`
`scipy, numpy, pandas, textblob`

---
