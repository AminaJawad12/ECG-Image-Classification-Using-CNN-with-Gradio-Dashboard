# 🫀 ECG Image Classification Using CNN with Gradio Dashboard

An end-to-end **Deep Learning–based ECG Image Classification system** using **Convolutional Neural Networks (CNN)**, integrated with an **interactive Gradio medical dashboard**. The system classifies ECG images into multiple cardiac conditions and provides **clinical interpretation, risk assessment, and recommendations**.

---

## 📌 Project Overview

Electrocardiograms (ECGs) are essential diagnostic tools for identifying heart-related conditions. This project applies **Computer Vision and Deep Learning** techniques to automatically classify ECG images into five distinct categories:

- **Myocardial Infarction (MI)**
- **History of Myocardial Infarction (HMI)**
- **Abnormal Heartbeat (AHB)**
- **Normal ECG**
- **Non-ECG Images**

A trained CNN model is deployed using **Gradio**, offering a user-friendly medical dashboard that displays predictions, confidence scores, and actionable clinical insights.

---

## 🚀 Key Features

- CNN-based ECG image classification
- Supports **5 ECG / Non-ECG classes**
- High classification accuracy (**~97%**)
- Interactive **Gradio medical dashboard**
- Confidence score & probability distribution visualization
- Automated **risk level assessment** (LOW / MEDIUM / HIGH)
- Clinical interpretation & recommendations
- Suggested medical tests & lifestyle advice
- Confusion matrix & classification report

---

## 🧠 Model Architecture

**Input Shape:** `224 × 224 × 3`

### CNN Layers

- Conv2D (32 filters) → MaxPooling
- Conv2D (64 filters) → MaxPooling
- Conv2D (128 filters) → MaxPooling

### Fully Connected Layers

- Dense (256 units) + Dropout (0.5)
- Dense (5 units) with **Softmax activation**

**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy

---

## 📂 Dataset

The dataset consists of labeled ECG images representing multiple cardiac conditions and non-ECG samples.

### Dataset Classes

- ECG Images of **Myocardial Infarction patients**
- ECG Images of patients with **History of MI**
- ECG Images showing **Abnormal Heartbeat**
- **Normal ECG** images
- **Non-ECG** images

### Dataset Link

📎 **Mendeley Dataset:**  
https://data.mendeley.com/datasets/gwbz3fsgp8/2

### Preprocessing Steps

- Resize all images to **224 × 224**
- Normalize pixel values
- Convert labels to categorical format

---

## 📊 Model Performance

- **Overall Accuracy:** ~97%
- **Macro F1-score:** ~0.96

### Confusion Matrix Summary

- Excellent separation between ECG classes
- Near-perfect classification for **Non-ECG images**
- Minimal misclassification among cardiac conditions

---

## 🖥️ Gradio Medical Dashboard

The Gradio-based interface allows users to:

- Upload an ECG image
- View the **predicted class & confidence score**
- Analyze probability distribution across all classes
- Receive automated **risk level assessment**
- Read **clinical interpretation** of the result
- Get recommended **medical tests**
- Receive **lifestyle & monitoring advice**
- View emergency guidance for high-risk predictions

---

## 📈 Result Screenshots

![Result Dashboard Screenshot 1](https://raw.githubusercontent.com/AminaJawad12/ECG-Image-Classification-Using-CNN-with-Gradio-Dashboard/ff11adafa61e6f72bee97e22362ff63e4f8cf488/result_dashboard_1.jpeg)

![Result Dashboard Screenshot 2](https://raw.githubusercontent.com/AminaJawad12/ECG-Image-Classification-Using-CNN-with-Gradio-Dashboard/ff11adafa61e6f72bee97e22362ff63e4f8cf488/result_dashboard_2.jpeg)

---

## 🧪 Technologies Used

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Scikit-learn**
- **Matplotlib & Seaborn**
- **Gradio**
- **Google Colab**

---

## ⚠️ Disclaimer

This project is intended **for educational and research purposes only**. It is **not a substitute for professional medical diagnosis or clinical decision-making**. Always consult a qualified healthcare professional for medical advice.

---

## ⭐ Acknowledgments

- ECG dataset provided via **Mendeley Data**
- Open-source libraries and tools that enabled model development and deployment

---

If you find this project useful, feel free to ⭐ star the repository and contribute!
