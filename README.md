# 🌟 **ClassiVision** - Live Train, Preview & Effortless Export 🖼️🤖

**ClassiVision** is an intuitive, web-based image classification tool designed to make training machine learning models fast, interactive, and effortless. Inspired by the simplicity of [Teachable Machine by Google](https://teachablemachine.withgoogle.com/), **ClassiVision** allows users to easily upload or capture images via webcam to create datasets for training custom image classification models.

## 🚀 Key Features:
- **📸 Capture or Upload Images**: Capture images using your webcam or upload images directly from your device to create a custom dataset for your classification task.
- **⚙️ Adjustable Training Parameters**: Choose model training parameters such as epochs, learning rate, batch size, etc., to tailor the model to your needs.
- **⏱️ Real-Time Training**: Train your custom model directly in the browser without the need for external tools or setups. Watch the training progress in real-time!
- **💡 Prediction On-The-Go**: Once your model is trained, predict the class of uploaded images within the app itself.
- **📥 Effortless Export**: Download your trained model in multiple formats like `.zip` or `.tflite` to use in real-world applications or mobile devices.

## 🎯 Project Aim:
**ClassiVision** is designed with the goal of providing an **interactive, easy-to-use interface** for image classification tasks. It empowers anyone, from beginners to experienced developers, to train and deploy custom models within minutes, using nothing more than a web browser.

## 🖥️ How It Works:
1. **Image Dataset Creation**: Choose to upload images or capture them using your webcam. Organize your dataset based on classes.
2. **Set Training Parameters**: Adjust model parameters like epochs, learning rate, and batch size for optimal performance.
3. **Train the Model**: Start training your model with just a few clicks! The app provides real-time updates on the training process.
4. **Make Predictions**: Once training is complete, use the model to predict the class of new images uploaded directly on the web app.
5. **Export the Model**: Download the trained model in `.zip` or `.tflite` format, ready for deployment.

## 📂 Directory Structure:
```plaintext
data/
  └── {class_name}/
        ├── img1.jpg
        ├── img2.jpg
        └── ...
models/
  ├── trained_model.h5
  └── trained_model.tflite
app.py
requirements.txt
