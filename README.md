# Detecting Engines in Audio Files with Machine Learning

<a href="https://colab.research.google.com/github/brayvid/illegal-logging-detection/blob/main/engine_detection.ipynb" rel="Open in Colab"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="" /></a>
<h4>Blake Rayvid - <a href=https://github.com/brayvid>https://github.com/brayvid</a></h4>
Flatiron School Data Science Bootcamp Phase 4 Project

<a href="https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50">Dataset 1</a>
<a href="https://www.kaggle.com/datasets/nicorenaldo/chainsaw-testing">Dataset 2</a>

This project focuses on developing a machine learning model to detect engine sounds in audio files. Two different approaches were used: an XGBoost classifier and a Convolutional Neural Network (CNN). The goal was to compare their performances and select the best model for deployment.

## Overview
The objective of this project is to build a robust classifier that can accurately identify engine sounds from audio files. The project includes two main models: an XGBoost classifier, which is a tree-based method, and a CNN, which is a deep learning model suitable for processing audio data represented as spectrograms.

## Data Preparation
The dataset consists of audio files labeled as either containing engine sounds or not. The audio files were preprocessed by converting them into spectrograms (for CNN) and by flattening the audio features (for XGBoost).

### Steps:
1. **Audio Preprocessing**:
   - Converted audio files to a uniform format (e.g., mono, 16kHz).
   - Extracted features like Mel-frequency cepstral coefficients (MFCCs) for the XGBoost model.
   - Generated spectrograms as 2D images for the CNN model.

2. **Data Splitting**:
   - Split the dataset into training and testing sets (90% training, 10% testing).

## Modeling Approaches

### XGBoost Classifier
The XGBoost model was trained using RandomizedSearchCV to find the best hyperparameters. The features were normalized using `StandardScaler`.

- **Model Configuration**:
  - `n_estimators`: Number of trees.
  - `max_depth`: Maximum depth of the trees.
  - `learning_rate`: Learning rate for boosting.
  - `tree_method`: Set to `hist` for histogram-based optimization.
  - `device`: Set to `cuda` for GPU acceleration.

- **Training**:
  - The model was trained on the preprocessed audio features.
  - RandomizedSearchCV was used to optimize hyperparameters.

### Convolutional Neural Network (CNN)
The CNN was designed to process spectrograms of audio files. The model architecture included several convolutional layers followed by pooling layers and fully connected layers.

- **Model Configuration**:
  - Several convolutional layers with ReLU activation.
  - MaxPooling layers to downsample the feature maps.
  - Fully connected layers leading to a final softmax output for classification.

- **Training**:
  - The model was trained using backpropagation and optimized using Adam.
  - Data augmentation techniques were applied to improve generalization.

## Evaluation
Both models were evaluated on the test set using metrics such as ROC-AUC, accuracy, precision, recall, and F1-score. The confusion matrix was also analyzed to understand the distribution of false positives and false negatives.

### Key Results:
- **XGBoost**:
  - ROC-AUC: 0.65
  - Other metrics showed the model struggled with certain types of engine sounds.

- **CNN**:
  - The CNN generally performed better on audio data due to its ability to capture complex patterns in spectrograms.
  - ROC-AUC and other metrics were typically higher compared to XGBoost.

## Next steps
- Gather more examples of engines.
- Try IMBlearn pipeline with SMOTE upsampling.
- Deploy to a cloud platform or use edge computing.
