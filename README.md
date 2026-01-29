# 🫀 ECG Image Classification Using CNN with Gradio Dashboard

An end-to-end Deep Learning–based ECG Image Classification system using Convolutional Neural Networks (CNN), integrated with an interactive Gradio medical dashboard. The system classifies ECG images into multiple cardiac conditions and provides clinical interpretation, risk assessment, and recommendations.

---

## 📌 Project Overview

Electrocardiograms (ECGs) play a crucial role in diagnosing heart-related conditions. This project leverages Computer Vision and Deep Learning to automatically classify ECG images into five categories: Myocardial Infarction, History of Myocardial Infarction, Abnormal Heartbeat, Normal ECG, and Non-ECG Images. The trained CNN model is deployed using Gradio, offering a user-friendly medical dashboard that displays predictions, confidence scores, and clinical insights.

---

## 🚀 Features

- CNN-based ECG image classification  
- Supports 5 different ECG / Non-ECG classes  
- High accuracy (~97%)  
- Interactive Gradio dashboard  
- Medical risk level & interpretation  
- Probability distribution visualization  
- Clinical test & lifestyle recommendations  
- Confusion matrix & classification report  

---

## 🧠 Model Architecture

**Input Size:** 224 × 224 × 3  

**CNN Layers:** Conv2D (32) + MaxPooling → Conv2D (64) + MaxPooling → Conv2D (128) + MaxPooling  

**Fully Connected Layers:** Dense (256) + Dropout (0.5) → Dense (5) with Softmax  

**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy  

---

## 📂 Dataset Structure

- ECG Images of Myocardial Infarction Patients  
- ECG Images of Patients with History of MI  
- ECG Images of Patients with Abnormal Heartbeat  
- Normal Person ECG Images  
- Non-ECG Images  

Each ECG image is resized to 224×224 and normalized before training.

---

## 📊 Model Performance

- Overall Accuracy: ~97%  
- Macro F1-score: ~0.96  

Confusion Matrix Summary: Excellent separation between ECG classes; near-perfect classification for Non-ECG images.

---

## 🖥️ Gradio Medical Dashboard

The Gradio interface allows users to upload an ECG image, view the predicted class & confidence, and receive risk level (LOW / MEDIUM / HIGH), clinical interpretation, recommended medical tests, lifestyle & monitoring advice, and emergency guidance.

---

## ⚠️ Disclaimer

This tool is for educational and research purposes only and **not a substitute for professional medical diagnosis**.

---

## 🧪 Technologies Used

Python, TensorFlow / Keras, OpenCV, NumPy, Scikit-learn, Matplotlib & Seaborn, Gradio, Google Colab  

---

## 📈 Result Screenshots

![Result Screenshot 1](https://raw.githubusercontent.com/AminaJawad12/ECG-Image-Classification-Using-CNN-with-Gradio-Dashboard/ff11adafa61e6f72bee97e22362ff63e4f8cf488/result_dashboard_1.jpeg)

![Result Screenshot 2](https://raw.githubusercontent.com/AminaJawad12/ECG-Image-Classification-Using-CNN-with-Gradio-Dashboard/ff11adafa61e6f72bee97e22362ff63e4f8cf488/result_dashboard_2.jpeg)

