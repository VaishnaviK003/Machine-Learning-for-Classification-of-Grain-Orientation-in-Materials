# Brick Grain Orientation Classification using CNN + Grain Detection

This project implements a **hybrid computer vision + deep learning pipeline** to automatically classify the dominant grain orientation in brick microstructures. It is intended for research work involving microscopy or industrial manufacturing analysis.

## Problem Statement
Brick microstructures often exhibit anisotropic (directional) grain patterns. These grains tend to align either **vertically (≈ 0°)** or **horizontally (≈ 90°)**. Classifying a brick based on this global orientation can assist in analyzing microstructural anisotropy, understanding manufacturing directionality, and linking structure to material properties.

## Solution Overview
This workflow combines **classical image processing** and **deep learning**: Grain detection uses thresholding, morphological filtering, and watershed segmentation to isolate grains. Orientation extraction fits ellipses on detected grains (`cv2.fitEllipse`) and computes angles. A histogram of angles determines a single dominant orientation label per brick. The dataset is then augmented through rotation and scaling directly on disk. A **ResNet50** classifier (transfer learning) is trained to predict whether a brick is predominantly **0° (vertical)** or **90° (horizontal)**. The script produces publication-ready outputs including confusion matrices, training curves, and labeled prediction images.

## Folder Structure (Auto-Generated)
`InputImages/` (unlabelled microscopy data); `data_split/train/`, `data_split/val/`, `data_split/test/` (after automatic splitting & augmentation); `data_split/classified_test_images/` (final labeled predictions); plus results such as `loss_curve.png`, `accuracy_curve.png`, `confusion_matrix_orientation.png`, `training_history.json`, and `resnet50_brick_orientation_model.keras`.

## Methods
**Grain Detection & Angle Measurement:** CLAHE contrast enhancement, bilateral filtering, adaptive + Otsu thresholding, morphological filtering, watershed segmentation, ellipse fitting, and orientation normalization to [-90°, 90°].  
**Dominant Orientation Rule:** `|angle| ≤ 15° → Vertical (label 0)` and `|angle| > 15° → Horizontal (label 1)`.  
**Deep Learning Model:** pretrained **ResNet50**, global average pooling and dense layers, loss = sparse categorical crossentropy, optimizer = Adam (1e-4), metrics = accuracy.  
**On-Disk Augmentation:** Rotation (±10°) and uniform scaling (0.8×, 1.2×).

## How to Run
1. Place raw microscopy images into `/content/InputImages/`.  
2. Run the provided script (recommended in Google Colab with GPU).  
The pipeline will automatically: split train/val/test, augment images, detect grains, label orientations, train the classifier, and output final results. **No manual labeling is required.**

## Outputs for Research / Papers
The script saves: `loss_curve.png` (training curve); `accuracy_curve.png` (validation curve); `confusion_matrix_orientation.png` (evaluation); and `classified_test_images/` (each test image overlaid with true/predicted orientation). These outputs are publication-ready and ideal for journal figures or thesis documentation.

## Requirements
Python ≥ 3.8, TensorFlow ≥ 2.10 (Colab recommended), OpenCV ≥ 4.6, NumPy, Matplotlib, and Scikit-Learn. Standard Colab environments already satisfy these requirements.

## Future Work
Potential extensions include multi-orientation prediction, fine-tuning the full ResNet50 backbone, improved grain segmentation, and support for classification of other anisotropic ceramics.

## Author Notes
This pipeline was developed for **material microstructure characterization** and can be adapted to other anisotropic materials such as fiber-reinforced ceramics, metals, and composites.

Acknowledgements:

Dr. Iman Soltani: Thank you for all the interesting lectures on machine learning which paved way for this project. 

Dr. Valeria La Saponara & ACRES Lab: Thank you for sharing your work on mycelium bricks and allowing for data collection for this project. 

Dr. Christina Cogdell & Team: Thank you for sharing your work on mycelium bricks which helped with understanding micro/macrostructural behavior.

Data & Results:
https://drive.google.com/drive/folders/1LNbNj1PwuwO5hYOSqTjDRiBqCtvYi-Ol?usp=sharing


