# Predective-Modelling-and-Natural-Language-Processing

This repository contains a machine learning pipeline designed to perform text classification and sentiment prediction. The project implements a natural language processing (NLP) workflow to process social media text, extract textual features, and train multiple classification models to categorize target sentiments.

##  Project Overview

The core objective of this project is to parse multi-platform textual logs, clean data anomalies, vectorize text data using advanced statistical extraction methods, and train models to handle complex sentiment tracking. 

### Data Profile
* **Dataset Size:** 732 rows, 15 columns
* **Missing Values:** 0 missing values across all structural fields
* **Input Feature:** `Text` (Social media snippets, posts, or reviews)
* **Target Classification:** `Sentiment` (Multi-class sentiment labels)

---

##  Dataset Structure & Preprocessing

The input file `sentiment.csv` contains text fields alongside temporal, regional, and engagement metadata:

* **Textual / Categorical Data:** `Text`, `User`, `Platform`, `Hashtags`, `Country`
* **Engagement Metrics:** `Retweets`, `Likes`
* **Temporal Attributes:** `Timestamp`, `Year`, `Month`, `Day`, `Hour`

### Preprocessing Steps Implemented:
1. **Redundancy Removal:** Dropped system-generated indexing columns (`Unnamed: 0`) to enforce lean data modeling.
2. **Target Label Encoding:** Converted the multi-class categorical `Sentiment` column into model-friendly integers using `LabelEncoder`.
3. **Data Quality Check:** Verified complete structural coverage with `df.isnull().sum()` ensuring a completely dense matrix before training.

---

##  Tech Stack & Dependencies

The pipeline is built using Python inside a cloud runtime environment (Google Colab) leveraging standard open-source scientific libraries:

* **Data Wrangling:** `pandas`
* **Visualizations:** `matplotlib`
* **Feature Engineering & Vectors:** `scikit-learn` (TfidfVectorizer)
* **Model Deployment Pipelines:** `scikit-learn` (Pipeline, GridSearchCV)
* **Core Machine Learning Models:**
  * Logistic Regression
  * Decision Tree Classifier
  * Random Forest Classifier

---

##  Machine Learning Pipeline Architecture

The implementation uses a robust decoupled design sequence to ensure zero data leakage between training slices:

Utilisez le code avec précaution.[ Raw Text Data ]│▼[ Train-Test Split (Holdout Validation) ]│▼[ TfidfVectorizer ] ──► (Converts raw string text into sparse TF-IDF matrices)│▼[ Pipeline Architectures ] ──► (Combines TF-IDF step seamlessly with Estimators)│▼[ GridSearch Hyperparameter Tuning ] ──► (Optimizes model parameters using Cross-Validation)
### Evaluated Model Metrics:
To ensure the multi-class model is highly stable and reliable, performance is validated across standard classification benchmarks:
* **Accuracy:** Total global prediction correctness rate.
* **Precision:** True prediction validity out of all assigned class tags.
* **Recall:** True detection coverage capacity of actual target items.
* **F1-Score:** Harmonic evaluation balance score pairing Precision and Recall.

---

##  Quick Start & Usage

To directly access and execute this notebook within Google Colab without manual local dataset file configurations, you can pull the raw data asset straight into your environment with the following initialization code snippet:

```python
import pandas as pd

# Direct streaming configuration from GitHub source tracking URL
url = "https://raw.githubusercontent.com/Arij0106/Predective-Modelling-and-Natural-Language-Processing/refs/heads/main/sentiment.csv"
df = pd.read_csv(url)

# Proceed to data processing pipeline execution
print(df.head())
```
