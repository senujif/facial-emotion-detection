# Facial Emotion Detection using CNN 🎭🧠

This project implements a real-time facial emotion recognition system using a custom Convolutional Neural Network (CNN) model trained on the FER-2013 dataset. The system detects facial expressions from a live webcam stream and classifies them into five categories: **Angry**, **Happy**, **Neutral**, **Sad**, and **Surprise**.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `main.py` | Core training and model saving script |
| `detect_emotion_live.py` | Real-time emotion detection using webcam (grayscale) |
| `evaluate_model.py` | Model evaluation on validation data and classification report |
| `confusion_matrix.py` | Generate confusion matrix heatmap |
| `Training vs validation loss curve` | Code to plot accuracy and loss graphs |
| `requirements.txt` | List of required Python libraries |

---

## 🔧 Technologies Used

- Python 3.6 
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib & Seaborn  
- Scikit-learn  

---

## ▶️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt

python detect_emotion_live.py
