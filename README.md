# 🫀 ECG Image Classification Using CNN with Gradio Dashboard

📎 **Dataset (Mendeley Data):**  
https://data.mendeley.com/datasets/gwbz3fsgp8/2  

An end-to-end **Deep Learning–based ECG Image Classification system** using **Convolutional Neural Networks (CNN)**, integrated with an **interactive Gradio medical dashboard**. The system classifies ECG images into multiple cardiac conditions and provides **clinical interpretation, risk assessment, and recommendations**.

---

## 📌 Project Overview

Electrocardiograms (ECGs) are widely used diagnostic tools for detecting and monitoring heart-related conditions. This project leverages **Computer Vision and Deep Learning** techniques to automatically classify ECG images into five clinically relevant categories:

- **Myocardial Infarction (MI)**
- **History of Myocardial Infarction (HMI)**
- **Abnormal Heartbeat (AHB)**
- **Normal ECG**
- **Non-ECG Images**

The trained CNN model is deployed using **Gradio**, providing a user-friendly medical dashboard that presents predictions, confidence scores, and clinically meaningful insights.

---

## 🚀 Key Features

- CNN-based ECG image classification
- Supports **5 ECG / Non-ECG classes**
- High classification accuracy (**~97%**)
- Interactive **Gradio medical dashboard**
- Prediction confidence & probability distribution visualization
- Automated **risk level assessment** (LOW / MEDIUM / HIGH)
- Clinical interpretation of results
- Recommended medical tests and lifestyle guidance
- Confusion matrix & detailed classification report

---

## 🧠 Model Architecture

**Input Shape:** `224 × 224 × 3`

### Convolutional Layers

- Conv2D (32 filters) → MaxPooling
- Conv2D (64 filters) → MaxPooling
- Conv2D (128 filters) → MaxPooling

### Fully Connected Layers

- Dense (256 units) + Dropout (0.5)
- Dense (5 units) with **Softmax activation**

**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy

---

## 📂 Dataset Description

The dataset consists of labeled ECG image samples representing multiple cardiac conditions along with non-ECG images for robust classification.

### Dataset Classes

- ECG images of **Myocardial Infarction patients**
- ECG images of patients with **History of MI**
- ECG images showing **Abnormal Heartbeat**
- **Normal ECG** images
- **Non-ECG** images

### Dataset Source

📎 **Mendeley Data:**  
https://data.mendeley.com/datasets/gwbz3fsgp8/2

### Data Preprocessing

- Images resized to **224 × 224**
- Pixel value normalization
- One-hot encoding of class labels

---

## 📊 Model Performance

- **Overall Accuracy:** ~97%
- **Macro F1-score:** ~0.96

### Confusion Matrix Insights

- Strong class separation across all ECG categories
- Near-perfect detection of **Non-ECG images**
- Minimal overlap between cardiac conditions

---

## 🖥️ Gradio Medical Dashboard

The Gradio-based dashboard enables users to:

- Upload an ECG image
- View the **predicted class and confidence score**
- Inspect probability distribution across all classes
- Receive an automated **risk level assessment**
- Read clinical interpretation of predictions
- Get recommended diagnostic tests
- Receive lifestyle and monitoring advice
- Access emergency guidance for high-risk cases

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

This project is intended **strictly for educational and research purposes**. It is **not a substitute for professional medical diagnosis or clinical decision-making**. Always consult qualified healthcare professionals for medical advice.

---

## ⭐ Acknowledgments

- ECG dataset provided via **Mendeley Data**
- Open-source libraries and frameworks that supported model development and deployment

---

If you find this project useful, feel free to ⭐ star the repository and contribute!
