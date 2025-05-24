# ASL_Detection
# 🤟 ASL (American Sign Language) Detection

This project demonstrates a deep learning-based approach to detecting and classifying ASL (American Sign Language) hand signs. The entire workflow was conducted on Google Colab and includes data preprocessing, model training, and real-time predictions.

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)

## 🧾 Overview

This project aims to build a deep learning model that can accurately recognize American Sign Language (ASL) alphabets using image data. This has the potential to aid communication for the hearing- and speech-impaired communities.

## 📊 Dataset

- **Source**: [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet)
- **Classes**: 29 (A-Z, plus `del`, `nothing`, and `space`)
- **Description**: 87,000+ labeled images of hand signs in 200x200 pixel JPEG format.

## 🧠 Approach

1. **Data Loading & Preprocessing**
   - Resizing images
   - One-hot encoding labels
   - Train-test split

2. **Model Building**
   - CNN architecture built using Keras with TensorFlow backend
   - Used data augmentation to improve generalization

3. **Training**
   - Trained with `Adam` optimizer and `categorical_crossentropy` loss
   - Early stopping to prevent overfitting

4. **Evaluation**
   - Accuracy, Confusion Matrix, Classification Report

## 📈 Model Performance

- **Validation Accuracy**: ~95%
- **Test Accuracy**: ~94%
- **Model**: CNN (3 Convolutional Layers + Dense Layers)
- **Metrics Used**: Accuracy, Precision, Recall

> The model performs well across most ASL signs with minimal confusion among similar-looking signs like `M`, `N`, and `T`.

## 🧰 Technologies Used

- Google Colab
- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas, Matplotlib, Seaborn
