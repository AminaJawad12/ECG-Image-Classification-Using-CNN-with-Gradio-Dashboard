# 🫀 ECG Image Classification Using CNN with Gradio Dashboard

An end-to-end Deep Learning–based ECG Image Classification system using Convolutional Neural Networks (CNN), integrated with an interactive Gradio medical dashboard.  
The system classifies ECG images into multiple cardiac conditions and provides clinical interpretation, risk assessment, and recommendations.

---

## 📌 Project Overview

Electrocardiograms (ECGs) play a crucial role in diagnosing heart-related conditions. This project leverages Computer Vision and Deep Learning to automatically classify ECG images into five categories:

- **Myocardial Infarction**  
- **History of Myocardial Infarction**  
- **Abnormal Heartbeat**  
- **Normal ECG**  
- **Non-ECG Images**  

The trained CNN model is deployed using Gradio, offering a user-friendly medical dashboard that displays predictions, confidence scores, and clinical insights.

---

## 🚀 Features

- ✅ CNN-based ECG image classification  
- ✅ Supports 5 different ECG / Non-ECG classes  
- ✅ High accuracy (~97%)  
- ✅ Interactive Gradio dashboard  
- ✅ Medical risk level & interpretation  
- ✅ Probability distribution visualization  
- ✅ Clinical test & lifestyle recommendations  
- ✅ Confusion matrix & classification report  

---

## 🧠 Model Architecture

**Input Size:** 224 × 224 × 3  

**CNN Layers:**  
- Conv2D (32) + MaxPooling  
- Conv2D (64) + MaxPooling  
- Conv2D (128) + MaxPooling  

**Fully Connected Layers:**  
- Dense (256) + Dropout (0.5)  
- Dense (5) with Softmax  

**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy  

---

## 📂 Dataset Structure

- ✅ ECG Images of Myocardial Infarction Patients
- ✅ ECG Images of Patient that have History of MI  
- ✅ ECG Images of Patient that have abnormal heartbeat
- ✅ Normal Person ECG Images
- ✅ nonecg_images

  ---

  
Each ECG image is resized to 224×224 and normalized before training.

---

## 📊 Model Performance

- **Overall Accuracy:** ~97%  
- **Macro F1-score:** ~0.96  

**Confusion Matrix Summary:**  
- Excellent separation between ECG classes  
- Near-perfect classification for Non-ECG images  

---

## 🖥️ Gradio Medical Dashboard

The Gradio interface allows users to:

- Upload an ECG image  
- View predicted class & confidence  

And receive:  

- Risk level (LOW / MEDIUM / HIGH)  
- Clinical interpretation  
- Recommended medical tests  
- Lifestyle & monitoring advice  
- Emergency guidance  

---

## ⚠️ Disclaimer

This tool is for educational and research purposes only and **not a substitute for professional medical diagnosis**.

---

## 🧪 Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Scikit-learn  
- Matplotlib & Seaborn  
- Gradio  
- Google Colab  

---
## 🚀 How to Run the Project (Google Colab)

Follow these steps to run the ECG Image Classification project in **Google Colab**:

```bash
# 1️⃣ Clone the repository
git clone https://github.com/your-username/ecg-image-classification-cnn.git
cd ecg-image-classification-cnn

# 2️⃣ Download and prepare the dataset
# Download the ECG dataset, extract it, and upload to Colab in the structure:
# dataset/
#  └── ECG_Images/

# 3️⃣ Open the notebook in Colab
# ECG_Image_Classification_CNN_Gradio.ipynb

# 4️⃣ Install dependencies
pip install tensorflow keras numpy opencv-python scikit-learn matplotlib seaborn gradio

# 5️⃣ Run the notebook
# Execute all cells sequentially to preprocess ECG images, train the CNN model, evaluate performance, and launch the Gradio dashboard.

---

## 📈 Result Screenshot

![Result Screenshot](https://github.com/AminaJawad12/Human-Action-Recognition-using-CNN-LSTM-on-the-UCF50-Dataset-with-Model-Evaluation-Dashboard/blob/main/Result%20image.jpeg?raw=true)

---

## ✅ Notes
- Training time depends on hardware availability.
- Using **Google Colab GPU** is strongly recommended.
- Always run all cells **in order** to avoid errors.

---

⭐ If you find this project helpful, consider giving the repository a star!



