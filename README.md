# 🎯 Radar Detection using YOLO

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-yellow)

## 📖 Overview
This project focuses on detecting and classifying target objects (such as **Cars** and **Buses** from radar/camera data) using a custom fine-tuned YOLO object detection model. It encompasses the complete deep learning pipeline—from dataset exploration and preprocessing to model training, evaluation, and inference.

---

## ✨ Features
- **Dataset Exploration:** Analyzing the distribution of vehicle classes (`Dataset_Exploration.ipynb`).
- **Data Preprocessing & Augmentation:** Preparing and augmenting the dataset for optimal training (`Data_Preprocessing_Augmentation.ipynb`).
- **Custom YOLO Fine-Tuning:** Leveraging PyTorch with GPU acceleration to train a custom YOLO model (`Model_Training.ipynb`).
- **Inference & Visualization:** Evaluating model performance and comparing the fine-tuned model against the base YOLO model.
- **Application:** An `app.py` script for easy and clean model deployment or testing.

---

## 📁 Project Structure
```text
Radar_Detection/
├── app.py                                  # Main application script for inference
├── Dataset_Exploration.ipynb               # EDA on the custom dataset
├── Data_Preprocessing_Augmentation.ipynb   # Data cleaning and augmentation
├── Model_Training.ipynb                    # YOLO model fine-tuning and evaluation
├── dataset_custom.yaml                     # YOLO dataset configuration
├── yolo26s.pt / yolo26n.pt                 # Base YOLO weights 
├── Dataset/ & Dataset_Split/               # Raw and split datasets (train/val/test)
└── runs/detect/radar_yolo_model/weights/   # Trained model weights (best.pt, last.pt)
```

---

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have Python installed, then install the required deep learning and computer vision dependencies. It is recommended to use a virtual environment:
```bash
pip install ultralytics torch torchvision opencv-python matplotlib
```

### 2. Pipeline Execution

1. **Understand your Data:** 
   Check out `Dataset_Exploration.ipynb` and `Data_Preprocessing_Augmentation.ipynb` to understand how the data is handled, split, and augmented.
2. **Train the Model:**
   Open `Model_Training.ipynb`. It automatically picks up `dataset_custom.yaml` and trains the network (defaults to 40 epochs on GPU if available).
3. **Run Inference:**
   You can use the resulting trained weights (found at `runs/detect/radar_yolo_model/weights/best.pt`) inside your `app.py` script to run predictions on new unseen data.

---

## 📊 Results Summary
The fine-tuned YOLO model is evaluated using **mAP50-95** metrics. The training notebook outputs a side-by-side visual comparison showing how the custom-trained radar detection model excels in identifying test images compared to the base pre-trained model.

---
