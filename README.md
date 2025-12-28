# Pneumonia Detection Using Deep Learning

## Project Description
This project focuses on automated pneumonia detection from chest X-ray (CXR) images using deep learning techniques. Pneumonia is a severe lung infection that requires timely diagnosis, but manual examination of X-rays can be inconsistent and error-prone. This system implements and compares three deep learning models for binary classification of CXR images into Normal or Pneumonic: Sequential CNN Model built from scratch using TensorFlow, VGG16 Autoencoder for feature extraction, and Keras Vision Transformer (ViT) leveraging transformer architecture. The project aims to evaluate the efficacy of these models in detecting pneumonia. The dataset used comprises 5856 anterior-posterior CXR images of children aged 1-5 years, categorized as Normal or Pneumonia.

## Features
- Binary classification of CXR images: Normal vs Pneumonia
- Comparative analysis of three models: Sequential CNN, VGG16 Autoencoder, Keras ViT
- Data preprocessing and augmentation to handle class imbalance and improve generalization
- Visualization of training/validation accuracy, loss curves, and confusion matrices
- Evaluation using metrics like test accuracy and ROC-AUC
- GPU-enabled training for faster computation
- Ready for future deployment on edge devices

## Dataset: https://data.mendeley.com/datasets/rscbjbr9sj/2
The dataset is organized into three main directories: train, val, and test. Each directory contains two subfolders: Normal and Pneumonia. Data augmentation is applied to increase dataset size and mitigate class imbalance, including rotation, flipping, scaling, and cropping.

## Model Details
Sequential CNN Model: Input size (224, 224, 3), 22 layers including Conv2D, SeparableConv2D, MaxPooling, BatchNormalization, Dropout, ReLU activation for hidden layers, Sigmoid for output, exponentially decaying learning rate with early stopping.
VGG16 Autoencoder: Input size (224, 224, 3), uses pre-trained VGG16 for feature extraction with softmax layer removed, fine-tuning performed on top layers.
Keras Vision Transformer (ViT): Input size (224, 224, 3), Multi-head self-attention and feed-forward layers for capturing global dependencies.
MobileViT Hybrid: Input 224x224x3, Conv Stem, Patch Extraction, Transformer, Global Average Pool.
MobileNetV3-Small: Input 224x224x3, MobileNet Conv Blocks, Global Average Pool.
EfficientNetV2-S: Input 224x224x3, EfficientNet Conv Blocks, Global Average Pool, Dropout. 

## Evaluation Metrics
- Accuracy: Measure of correctly predicted labels
- Confusion Matrix: Visualizes True/False Positives/Negatives
- ROC-AUC: Measures model’s ability to distinguish classes

Experimental results:
Sequential CNN: Test Accuracy 0.780, ROC-AUC 0.82
VGG16 Autoencoder: Test Accuracy 0.835, ROC-AUC 0.86
Keras ViT: Test Accuracy 0.716, ROC-AUC 0.78
MobileViT Hybrid: Test Accuracy 0.8141, ROC-AUC 0.9216
MobileNetV3S: Test Accuracy 0.7376, ROC-AUC 0.7638
EfficientNet V2S: Test Accuracy 0.7916, ROC-AUC 0.8254

## References
- Rajpurkar et al., CheXNet: Radiologist-level pneumonia detection, 2017
- LeCun et al., Deep Learning, Nature 2015
- Uparkar et al., Vision Transformer Outperforms CNN-based Model, 2023

## Future Work
- Expand to multi-class classification of thoracic diseases
- Explore hybrid CNN-ViT models for improved accuracy
- Deploy as a lightweight TFLite model for edge devices like Raspberry Pi
