# Comparing AlexNet CNN to SVM with ORB Feature Detector for CIFAR-10 Image Classification

Student Name: Fernando Nogueira  
Student ID: 300 195 829  
Date: October 31, 2025

## Broad Project Area

Computer Vision

## Problem Statement

This project compares a CNN's automatic feature extraction to a classical SVM baseline with handcrafted ORB features, quantifying the performance gap in reducing misclassifications.

## Dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

Link: https://www.cs.toronto.edu/~kriz/cifar.html

## Algorithm(s)

An AlexNet-inspired CNN adapted for CIFAR-10 and an SVM classifier with ORB (Oriented FAST and Rotated BRIEF) extracted features as a baseline.

## Model Evaluation

- Use an 80/10/10 train/validation/test split (from the 50k train set). 
- Tune hyperparams with Optuna on the validation set.
- Report test-set accuracy and F1-score for both models.
- Present results with confusion matrices and train/validation curves to compare the overfitting behaviors of the CNN.