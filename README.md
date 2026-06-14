# 🎭 Core Emotion Analytics System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Scikit--Learn-orange)](https://scikit-learn.org/)

A production-ready Natural Language Processing (NLP) pipeline built to analyze, parse, and classify the underlying emotional dimensions of textual data. This system transitions raw text sequences into structural feature vectors, achieving a high-performance validation baseline.

---

## 🚀 Live Interactive Interface
The repository includes a modern, high-fidelity dark-mode web dashboard built entirely via **Streamlit** for real-time model evaluation and vector distribution inference.

### 🖼️ UI Interface Preview
* Clean Metric Scoreboards mapping classification confidence.
* Real-time continuous output value distributions via normalized progress arrays.
* Fully styled custom CSS themes to deliver an executive-level portfolio layout.

---

## 🎯 Model & System Performance
While traditional lexical architectures struggle with context, this pipeline is optimized to extract structural patterns across **6 distinct semantic emotion classes**:

| Emotion Class | Icon Marker | Representation Mapping |
| :--- | :---: | :--- |
| **Sadness** | 😢 | Melancholy, Grief, Despair |
| **Joy** | 😊 | Happiness, Celebration, Relief |
| **Love** | 🥰 | Affection, Intimacy, Warmth |
| **Anger** | 😡 | Frustration, Resentment, Fury |
| **Fear** | 😨 | Anxiety, Panic, Apprehension |
| **Surprise** | 😲 | Astonishment, Shock, Wonder |

### 📊 Metric Validation Baseline
* **Validation Accuracy:** `88.78%` (~90% Framework Baseline)
* **Strategy:** Multi-Class Softmax Optimization
* **Feature Extraction Array:** Term Frequency-Inverse Document Frequency (TF-IDF Matrix)

---

## 🛠️ Technical Implementation Pipeline

### 1. Preprocessing Framework
Raw text entries pass through a meticulous linguistics cleanup phase before token serialization:
* Standardized lowercase text parsing.
* Advanced Word Tokenization mapping.
* NLTK **WordNet Lemmatization** to normalize verbs and root words smoothly.

### 2. Feature Vector Engineering
Instead of standard bag-of-words counting, features are engineered using an optimized `TfidfVectorizer`:
* **N-gram Range:** `(1, 2)` or `(1, 3)` configurations to capture key phrase negations (e.g., *"not happy"*).
* **Sublinear Scaling:** Logarithmic frequency compression (`sublinear_tf=True`) to prevent high-frequency words from dominating decision fields.

### 3. Classification Core Engine
A fast, memory-efficient **Multi-Class Logistic Regression** classifier maps regularized decision boundaries across the vector spaces using a balanced class-weight matrix.

---
