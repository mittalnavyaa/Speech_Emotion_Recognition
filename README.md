# 🎙️ Speech Emotion Recognition Project

## 📖 Overview
This repository contains code for **Speech Emotion Recognition (SER)** using a pre-trained wav2vec2 model and three popular emotional speech datasets: **RAVDESS**, **MELD**, and **CREMA-D**. The project processes raw audio files, extracts features using a pre-trained model, and performs emotion classification.

---

## 📂 Datasets Used

### 🔹 RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- Download: [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)
- Contains recordings from 24 professional actors (12 male, 12 female)
- 2 lexically-matched statements spoken with:
  - **8 emotions**: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`

### 🔹 MELD (Multimodal EmotionLines Dataset)
- Download: [https://www.kaggle.com/datasets/zaber666/meld-dataset](https://www.kaggle.com/datasets/zaber666/meld-dataset)
- Dialogue utterances from the TV series *Friends*
- **7 emotions**: `anger`, `disgust`, `sadness`, `joy`, `neutral`, `surprise`, `fear`

### 🔹 CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
- Download: [https://www.kaggle.com/datasets/ejlok1/cremad](https://www.kaggle.com/datasets/ejlok1/cremad)
- 7,442 clips from 91 actors
- **6 emotions**: `anger`, `disgust`, `fear`, `happy`, `neutral`, `sad`

---

## 🤖 Pre-trained Model
We use the [**ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition**](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) model from Hugging Face, which is a large-scale wav2vec2 model fine-tuned for emotion recognition from speech.

---

## 📊 Results

| Dataset   | Best Accuracy (%) | Final Test Accuracy (%) | Test Loss |
|-----------|-------------------|--------------------------|-----------|
| RAVDESS   | 94.10             | 93.75                    | 0.2225    |
| CREMA-D   | 70.38             | 55.47                    | 0.6943    |
| MELD      | 61.26             | 52.79                    | 1.2936    |

> 📌 *Best Accuracy* refers to the highest test accuracy observed across all epochs or folds during training.

---

## 📜 Files Description

### 🔧 Preprocessing Scripts
- `preprocess_RAVDESS.py` – Processes RAVDESS dataset into model-ready features
- `preprocess_MELD.py` – Processes MELD dataset audio
- `preprocess_CREMAD.py` – Processes CREMA-D audio clips
- `preprocess_MovieScope.py` – Prepares audio for genre/emotion correlation

### 🔍 Testing Scripts
- `test_RAVDESS.py` – Evaluates SER model on RAVDESS
- `test_MELD.py` – Evaluates model on MELD
- `test_CREMAD.py` – Evaluates model on CREMA-D

### 🛠️ Additional Scripts
- `Moviescope_wav.py` – Utility functions for audio processing
- `Moviescope_results_extraction.py` – Helper for result analysis

---

## 📦 Requirements

Install dependencies using pip:

```bash
pip install torch transformers librosa numpy pandas tqdm
