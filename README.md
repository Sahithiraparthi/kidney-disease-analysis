# kidney-disease-analysis
CNN-based kidney disease classification using CT images
# Kidney Disease Analysis Using CNN

A deep learning-based kidney disease classification project using Convolutional Neural Networks (CNN) and CT kidney images.

## Project Overview

This project uses CT kidney images to classify kidney conditions into four categories:

- Normal
- Cyst
- Tumor
- Stone

The images are preprocessed and passed through a Convolutional Neural Network to learn visual patterns and classify the kidney images.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset

The project uses the:

**CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone**

The dataset contains 12,446 CT kidney images.

The dataset was divided into:

- Training: 11,200 images
- Testing: 626 images
- Validation: 620 images

The dataset is not included in this repository because of its size.

## Image Preprocessing

- Images resized to 200 × 200 pixels
- Images converted to grayscale
- Pixel values normalized
- Dataset divided into training, validation, and testing sets

## CNN Model Architecture

The model consists of multiple convolutional and pooling layers followed by fully connected layers.

Architecture:

- Conv2D – 32 filters
- MaxPooling
- Conv2D – 32 filters
- MaxPooling
- Conv2D – 64 filters
- MaxPooling
- Conv2D – 64 filters
- MaxPooling
- Conv2D – 128 filters
- MaxPooling
- Conv2D – 128 filters
- MaxPooling
- Flatten
- Dense – 512 neurons
- Output layer – 4 classes

## Model Training

The model was trained for 5 epochs using:

- Optimizer: RMSprop
- Loss Function: Categorical Crossentropy
- Input Size: 200 × 200 grayscale images

## Results

The model achieved approximately:

| Metric | Score |
|---|---:|
| Accuracy | 99.36% |
| Precision | 98.87% |
| Recall | 99.38% |
| F1 Score | 99.11% |

## Project Structure

```text
kidney-disease-analysis/
│
├── kidney_disease_cnn.py
├── requirements.txt
├── README.md
└── .gitignore
