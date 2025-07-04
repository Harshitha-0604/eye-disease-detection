# eye-disease-detection
Built a lightweight CNN with TensorFlow Lite to classify retinal diseases from fundus images. Deployed a simple Streamlit web app for real-time image upload and prediction with minimal computational resources.

# 🧠 Eye Disease Detection using InceptionV3 (TensorFlow Lite)

A deep learning model that classifies retinal eye diseases from fundus images using **InceptionV3** and **TensorFlow Lite**. This lightweight solution supports real-time predictions even on low-compute devices, enabling broader accessibility for early diagnosis.

## 📌 Project Overview

This project aims to detect and classify eye diseases from retinal images using a CNN model:

- ✅ Built using **Transfer Learning** with InceptionV3
- ✅ Achieved **95% validation accuracy**
- ✅ Optimized using **TensorFlow Lite** for efficient deployment
- ✅ Supports deployment with Streamlit for real-time predictions (optional)

## 🧠 Model Architecture

- **Base Model:** InceptionV3 (pretrained on ImageNet)
- **Additional Layers:** Global Average Pooling, Dense, Dropout, and Output Softmax layer
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Output:** 4 Classes (e.g., Cataract, Glaucoma, Diabetic Retinopathy, Normal)

## 📂 Project Structure

📁 eye-disease-detection
├── eye-disease-detection-inceptionv3.ipynb # Jupyter notebook for training/testing
├── model.tflite # Final trained and optimized model
├── README.md # Project overview and documentation
├── LICENSE # MIT License
└── .gitignore # Ignored files


---

## 📊 Dataset

- **Type:** Retinal fundus images
- **Classes:** Cataract, Diabetic Retinopathy, Glaucoma, Normal
- **Image Size:** 224x224 RGB
- **Source:** [Insert dataset name or URL here]
- **Split:** 80% Training, 20% Validation

> *Note: The dataset used was for academic purposes. Replace the source above if public.*

---

## 🚀 How to Run

### 🔧 Install Dependencies
bash
pip install tensorflow opencv-python matplotlib
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

###Image Preprocessing and Prediction Instructions
import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    input_data = np.expand_dims(img, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Class mapping (update this based on your labels)
    classes = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
    predicted_class = classes[np.argmax(output)]

    print(f"Predicted Class: {predicted_class}")

# Example usage
predict_image("sample_image.jpg")

## 📷 Sample Output

The image below shows a sample prediction made by the trained model.

![Model Prediction](sample_output.png)

*Figure: Model prediction output using the TFLite version of InceptionV3.*




