# AI Teachable Machine using FastAPI & MobileNetV3

## Project Report

---

# 1. Introduction

The AI Teachable Machine project is a deep learning based image classification system inspired by Google Teachable Machine. The purpose of this project is to allow users to create their own custom AI image classifier without writing complex machine learning code.

Users can:

* Create custom classes dynamically
* Upload images for each class
* Train an AI model
* Predict uploaded images
* Predict webcam captured images
* Download trained AI model
* Monitor training progress live

The system uses:

* FastAPI for backend APIs
* Streamlit for frontend UI
* MobileNetV3 for transfer learning
* Logistic Regression classifier
* TensorFlow/Keras for deep learning

---

# 2. Objectives

The main objectives of the project are:

1. Build a user-friendly AI training platform
2. Allow dynamic image dataset creation
3. Train image classification models using transfer learning
4. Perform real-time image prediction
5. Provide professional frontend and backend architecture
6. Implement modern AI technologies
7. Create a scalable and reusable machine learning system

---

# 3. Technologies Used

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Main Programming Language      |
| FastAPI      | Backend API Development        |
| Streamlit    | Frontend User Interface        |
| TensorFlow   | Deep Learning Framework        |
| MobileNetV3  | Feature Extraction Model       |
| Scikit-learn | Logistic Regression Classifier |
| NumPy        | Numerical Computation          |
| Pillow (PIL) | Image Processing               |
| Uvicorn      | FastAPI Server                 |
| Requests     | API Communication              |

---

# 4. System Architecture

The project architecture is divided into two main parts:

## Frontend

The frontend is developed using Streamlit.

Responsibilities:

* User Interface
* Adding Classes
* Uploading Dataset Images
* Starting Training
* Displaying Accuracy & Loss
* Prediction Interface
* Webcam Prediction
* Download Trained Model

## Backend

The backend is developed using FastAPI.

Responsibilities:

* Dataset Management
* Image Upload APIs
* Training APIs
* Prediction APIs
* Model Saving & Loading
* Feature Extraction using MobileNetV3

---

# 5. Project Folder Structure

```plaintext
Hackathon Model/
│
├── backend/
│   ├── app.py
│   ├── dataset/
│   ├── model/
│   ├── temp/
│
├── frontend/
│   ├── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 6. Features of the Project

## Dynamic Class Creation

Users can create unlimited custom classes.

Examples:

* Cat
* Dog
* Car
* Mobile
* Bottle

Each class automatically creates a dataset folder.

---

## Multiple Image Upload

Users can upload multiple images for each class.

Features:

* JPG support
* PNG support
* JPEG support
* Multiple file upload
* Automatic file saving

---

## MobileNetV3 Transfer Learning

The project uses MobileNetV3 pretrained on ImageNet.

Advantages:

* Faster training
* Better accuracy
* Reduced computation
* Professional deep learning approach

Transfer learning helps the model learn from previously trained image features.

---

## Logistic Regression Classifier

After feature extraction using MobileNetV3, Logistic Regression is used as the final classifier.

Benefits:

* Lightweight
* Fast prediction
* Good performance on extracted features

---

## Live Training Dashboard

During training, the frontend shows:

* Accuracy
* Loss
* Epoch progress
* Progress bar
* Live charts

This improves user experience and model monitoring.

---

## Prediction System

The project supports:

### Upload Prediction

Users upload images and receive:

* Predicted class
* Confidence percentage

### Webcam Prediction

Users capture images directly using webcam.

---

## Download Trained Model

Users can download the trained AI model after training.

Supported format:

* .pkl model file

---

# 7. Backend API Endpoints

## Home Route

```python
GET /
```

Purpose:

Checks whether backend is running.

---

## Upload Sample API

```python
POST /upload-sample
```

Purpose:

Uploads dataset images for a specific class.

Parameters:

* class_name
* files

---

## Train API

```python
POST /train
```

Purpose:

Trains AI model using uploaded dataset.

---

## Predict API

```python
POST /predict
```

Purpose:

Predicts uploaded image.

Returns:

* Predicted class
* Confidence score

---

# 8. Training Process

The training process works in the following steps:

1. User uploads dataset images
2. Images are saved in class folders
3. MobileNetV3 extracts image features
4. Features are converted into vectors
5. Logistic Regression model is trained
6. Model is saved as .pkl file
7. Classes are saved separately

---

# 9. Image Preprocessing

Before training and prediction:

* Images are resized to 224x224
* Images are converted to RGB
* Pixel values are normalized
* MobileNetV3 preprocessing is applied

---

# 10. Frontend Design

The frontend contains:

* Modern dark UI
* Gradient backgrounds
* Responsive cards
* Styled buttons
* Upload sections
* Live metrics
* Prediction panels
* Webcam integration
* Download button

Custom CSS was used to create a professional appearance.

---

# 11. Error Handling

The system handles multiple errors:

* Empty dataset
* Invalid files
* Missing model
* Missing classes
* Upload errors
* Prediction errors
* Training errors

All errors are returned using JSON responses.

---

# 12. Advantages of the System

1. Easy to use
2. Dynamic AI training
3. No coding required for users
4. Professional UI
5. Real-time predictions
6. Fast training using transfer learning
7. Supports webcam prediction
8. Scalable architecture

---

# 13. Limitations

1. Requires enough training images
2. Large datasets may increase training time
3. Prediction accuracy depends on dataset quality
4. Currently supports image classification only

---

# 14. Future Improvements

Future enhancements may include:

* Real-time video prediction
* Cloud deployment
* Database integration
* User authentication
* Advanced deep learning models
* Multi-user support
* GPU acceleration
* Training history storage
* Better analytics dashboard

---

# 15. Conclusion

The AI Teachable Machine project successfully demonstrates the implementation of a modern AI-powered image classification platform using FastAPI, Streamlit, and MobileNetV3.

The system allows users to dynamically create datasets, train custom AI models, and perform real-time image predictions through an easy-to-use interface.

By combining transfer learning with a professional frontend and backend architecture, the project provides an efficient and scalable machine learning solution.

---

# 16. Run Commands

## Backend Run Command

```bash
C:\Users\DELL\AppData\Local\Programs\Python\Python39\python.exe -m uvicorn app:app --reload
```

## Frontend Run Command

```bash
streamlit run app.py
```

---

# 17. GitHub Commands

```bash
git init

git add .

git commit -m "AI Teachable Machine Project"

git branch -M main

git remote add origin https://github.com/USERNAME/REPOSITORY.git

git push -u origin main
```

---

# 18. Author

Project Developed By:

Sajid Hussain

AI / Machine Learning Project
