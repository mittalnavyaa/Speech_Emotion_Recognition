# Speech Emotion Recognition Project

## Overview
This repository contains code for Speech Emotion Recognition (SER) using pre-trained models and three popular emotional speech datasets: RAVDESS, MELD, and CREMAD. The project processes audio files, extracts features using a pre-trained wav2vec2 model, and performs emotion classification.

## Datasets

### RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- Download from: [Zenodo link](https://zenodo.org/record/1188976#.YO6yI-gzaUk)
- Contains 24 professional actors (12 male, 12 female) vocalizing two lexically-matched statements
- Includes 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised

### MELD (Multimodal Emotion Lines Dataset)
- Contains dialogue utterances from the TV series Friends
- Includes 7 emotions: anger, disgust, sadness, joy, neutral, surprise, fear

### CREMAD (Crowd-sourced Emotional Multimodal Actors Dataset)
- Contains 7,442 clips from 91 actors
- Includes 6 emotions: anger, disgust, fear, happy, neutral, sad

## Pre-trained Model
We use the `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` model from Hugging Face, which is a large-scale wav2vec2 model pre-trained for speech emotion recognition.

## Files Description

### Preprocessing Scripts
1. `preprocess_RAVDESS.py` - Processes raw RAVDESS audio files into features for training
2. `preprocess_MELD.py` - Processes MELD dataset audio files
3. `preprocess_CREMAD.py` - Processes CREMAD dataset audio files

### Testing Scripts
1. `test_RAVDESS.py` - Tests emotion recognition performance on RAVDESS dataset
2. `test_MELD.py` - Tests emotion recognition performance on MELD dataset
3. `test_CREMAD.py` - Tests emotion recognition performance on CREMAD dataset

### Additional Scripts
1. `Moviescope_wav.py` - Contains utility functions for audio processing and feature extraction

## Requirements
```bash
pip install torch transformers librosa numpy pandas tqdm
