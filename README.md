# 🇮🇳 Indian Fake News Detector

An AI-powered machine learning application that analyzes Indian news statements and predicts whether they are more likely to be **TRUE**, **FAKE**, or **UNCERTAIN**.

The project uses **TF-IDF text feature extraction** with a **Calibrated Linear SVM** classifier and provides an interactive Streamlit web application.

---

## 🚀 Live Demo

**Streamlit App:**  
PASTE_YOUR_STREAMLIT_URL_HERE

---

## 📌 Problem Statement

Misinformation spreads rapidly through online news and social media, making it difficult for students and general users to distinguish between credible and potentially misleading information.

This project aims to provide a machine-learning-based tool that analyzes the linguistic patterns of Indian news statements and provides a prediction with probability estimates.

> **Important:** This application is a machine-learning prediction tool, not a definitive fact-checking service. Important claims should always be verified using reliable sources.

---

## 🎯 Objectives

- Build a machine-learning model for Indian fake-news classification.
- Analyze news text using Natural Language Processing techniques.
- Compare multiple machine-learning algorithms.
- Select the best-performing classifier.
- Provide probability estimates for TRUE and FAKE predictions.
- Deploy the trained model as a web application.
- Provide an easy-to-use interface for students and users.

---

## 📊 Dataset

### IFND — Indian Fake News Dataset

The project uses the **Indian Fake News Dataset (IFND)**.

### Original dataset

- Total records: **56,714**
- Features: **7**
- Target column: `Label`
- Main text column: `Statement`

### Dataset columns

| Column | Description |
|---|---|
| `id` | Unique article identifier |
| `Statement` | News statement/text |
| `Image` | Image URL |
| `Web` | News source |
| `Category` | News category |
| `Date` | Publication date |
| `Label` | TRUE / FAKE target |

### Data cleaning

The original dataset contained duplicate statements and inconsistent label formatting.

After preprocessing:

- Clean records: **56,149**
- Duplicate statements remaining: **0**
- Missing Statement values: **0**
- Missing Label values: **0**

---

## 🧠 Machine Learning Pipeline

```text
Indian News Dataset
        ↓
Data Cleaning
        ↓
Duplicate Removal
        ↓
Train/Test Split
        ↓
TF-IDF Feature Extraction
        ↓
Model Training
        ↓
Model Comparison
        ↓
Probability Calibration
        ↓
Streamlit Application
