# Facial Emotion Detection using CNN 🎭🧠

This project implements a **real-time facial emotion recognition system** using a custom **Convolutional Neural Network (CNN)**. It classifies facial expressions into five categories: **Angry**, **Happy**, **Neutral**, **Sad**, and **Surprise**. The model is trained on the FER-2013 dataset and deployed in real-time using OpenCV and a webcam.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `main.py` | Full training pipeline (model architecture, data augmentation, training) |
| `detect_emotion_live.py` | Real-time emotion detection using webcam |
| `evaluate_model.py` | Evaluates model with validation accuracy and classification report |
| `confusion_matrix.py` | Generates confusion matrix heatmap |
| `Training vs validation loss curve` | Plots training/validation accuracy and loss |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## ▶️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Real-Time Emotion Detection
```bash
python detect_emotion_live.py
```
> Press `q` to quit the webcam window.

### 3. Train the Model (optional)
```bash
python main.py
```

### 4. Evaluate the Model
```bash
python evaluate_model.py
```

---

## 🧠 Model Summary

- **Architecture**: CNN with:
  - Conv2D, LeakyReLU, BatchNormalization, Dropout
  - GaussianNoise, L2 Regularization
- **Input Shape**: 48×48 grayscale images `(48, 48, 1)`
- **Output Classes**: Angry, Happy, Neutral, Sad, Surprise
- **Loss Function**: CategoricalCrossentropy with label smoothing
- **Optimizer**: Adam
- **Accuracy**: ~73–75% on validation set
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

<details>
  <summary>Click to view model architecture</summary>

```
Layer (type)                   Output Shape              Param #  
=================================================================
Conv2D → LeakyReLU             (None, 48, 48, 64)         ~640    
BatchNorm → MaxPool2D          (None, 24, 24, 64)         ...     
... (multiple layers)
Flatten → Dense → Dropout      (None, 512)               ...     
Dense (Softmax)                (None, 5)                 Final output
=================================================================
Total parameters: ~1.2M
```

</details>

---

## 📦 Dataset

- **FER-2013 Dataset**  
  Download from: [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

> Grayscale 48x48 facial images with 7 classes.  
> This project uses 5 classes: **Angry, Happy, Neutral, Sad, Surprise**

---

## 📊 Evaluation & Metrics

- ✅ Validation Accuracy: ~73–75%  
- 📉 Validation Loss: Output from `evaluate_model.py`  
- 📊 Classification Report: Precision, Recall, F1-score  
- 🔷 Confusion Matrix: Visualized with Seaborn (`confusion_matrix.py`)

---

## 🎯 Features

- Real-time facial emotion recognition via webcam
- CNN architecture with regularization and augmentation
- Live face detection using Haar cascades
- Deployable on standard hardware (no GPU required)

---

## 🧪 Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
tensorflow
opencv-python
numpy
matplotlib
seaborn
scikit-learn
```

---

## 📚 Report Reference

This repository contains the source code for:
**Appendix 8.1.1 – 8.1.4**  
as part of the academic project titled:

> **"Facial Emotion Detection Using Machine Learning"**

Includes:
- Real-time detection script
- Evaluation and metrics
- Confusion matrix visualization
- Full model training pipeline

---

## 📄 License

This project is provided for **academic and educational purposes**.  
You may reuse or adapt the code with appropriate credit.

---

## 🔗 GitHub Repository

> [https://github.com/senujif/facial-emotion-detection](https://github.com/senujif/facial-emotion-detection)
