# 🖼️ CNN Object Detection Mini Project

## 📌 Overview
This project implements **object detection using Convolutional Neural Networks (CNNs)**. It trains a **MobileNetV2-based model** on the **Oxford-IIIT Pet Dataset** to classify and detect objects in images. The performance is evaluated using **IoU (Intersection over Union) and mAP (Mean Average Precision).**

## 📂 Project Structure
📂 cnn_object_detection
│── 📂 dataset
│   ├── test_image.jpg  # Sample test image for evaluation
│── 📂 models
│   ├── pet_detector.h5  # Saved trained model
│── 📂 scripts
│   ├── load_dataset.py  # Load and visualize dataset
│   ├── preprocess_data.py  # Preprocess dataset (resize, normalize)
│   ├── cnn_model.py  # Define CNN model (MobileNetV2)
│   ├── train.py  # Train model
│   ├── evaluate.py  # Test model on new image
│   ├── iou_metric.py  # Compute IoU (Intersection over Union)
│   ├── map_metric.py  # Compute mAP (Mean Average Precision)
│── 📂 notebooks
│   ├── analysis.ipynb  # Jupyter Notebook for results visualization
│── 📂 results
│   ├── output_predictions.png  # Predicted images with bounding boxes
│   ├── model_comparison.csv  # Performance comparison of models
│── .gitignore
│── requirements.txt
│── README.md

## 📥 Dataset
- **Oxford-IIIT Pet Dataset** from TensorFlow Datasets.

## 🚀 Installation
1️⃣ **Clone Repository**
```bash
git clone https://github.com/Sahil9192/cnn_object_detection.git
cd cnn_object_detection
pip install -r requirements.txt
python scripts/train.py
python scripts/evaluate.py
