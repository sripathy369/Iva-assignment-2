# 🔭 VisionScope — Image & Video Analytics

> A modular computer vision application built with Python, OpenCV, Streamlit, and deep-learning based facial analysis.

🌐 **Live Application:**  
https://iva-assignment-2-fbprrzcabrsdmwq5qxqblh.streamlit.app/

---

## 📌 Project Overview

VisionScope is an interactive Image and Video Analytics application designed to demonstrate practical computer vision techniques through a unified web-based interface.

The application combines classical computer vision algorithms with modern face-analysis techniques and provides an intuitive dashboard for performing image inspection, face detection, template matching, video analysis, and analytical reporting.

Rather than implementing each computer vision technique as an isolated script, the project is organized into modular components so that individual processing tasks can be developed, tested, and extended independently.

---

## 🎯 Objectives

The primary objectives of this project are:

- To implement practical computer vision techniques using Python.
- To perform image preprocessing and statistical analysis.
- To detect human faces using the Viola-Jones approach.
- To locate specific visual patterns using template matching.
- To demonstrate FaceNet-based facial embeddings.
- To integrate DeepFace for facial analysis.
- To process uploaded images and videos through a web interface.
- To provide camera-based image input.
- To organize computer vision functionality into reusable modules.
- To deploy the application as an accessible web application using Streamlit.

---

## 🧠 Computer Vision Techniques

### 1. Image Analysis & Preprocessing

The Image Inspector provides basic image analytics and preprocessing operations.

Implemented operations include:

- Image dimension analysis
- Channel detection
- Brightness estimation
- Sharpness estimation
- Grayscale conversion
- Gaussian smoothing
- CLAHE-based contrast enhancement
- Canny edge detection

These operations demonstrate how raw image data can be transformed into representations that are more suitable for computer vision analysis.

---

### 2. Viola-Jones Face Detection

The Face Lab includes classical face detection using the Viola-Jones framework through OpenCV Haar Cascade classifiers.

The detector identifies potential human faces within an image and draws bounding boxes around detected regions.

The implementation uses OpenCV's pre-trained:

`haarcascade_frontalface_default.xml`

This demonstrates a classical object-detection approach that does not require a deep neural network for basic face localization.

---

### 3. Template Matching

The Pattern Finder module implements template matching using OpenCV's:

`cv2.matchTemplate()`

The system compares a smaller template image against a larger target image and determines the region with the highest matching score.

A threshold can be adjusted to control the sensitivity of the matching process.

This technique is useful for controlled visual inspection tasks where the target pattern has relatively consistent appearance and scale.

---

### 4. FaceNet Embeddings

The Face Lab also supports FaceNet-based facial representation.

Instead of treating a face simply as an image, FaceNet converts facial information into a numerical embedding vector.

These embeddings can then be compared using cosine similarity.

Conceptually:

**Face Image → FaceNet → Embedding Vector → Similarity Comparison**

This allows the application to demonstrate the concept of facial verification based on feature representations rather than direct pixel comparison.

---

### 5. DeepFace Analysis

DeepFace is integrated as an optional deep-learning component for facial analysis.

The application can perform emotion analysis and return the dominant detected emotion along with the corresponding analysis scores.

The deep-learning functionality is isolated from the core OpenCV functionality so that the application remains modular and easier to deploy.

---

## 🖥️ Application Modules

The application is divided into multiple functional areas:

### 🔹 Overview

Provides an introduction to the application and its available computer vision modules.

### 🔹 Image Inspector

Used for:

- Image statistics
- Preprocessing
- Edge detection
- Contrast enhancement
- Image quality inspection

### 🔹 Face Lab

Used for:

- Viola-Jones face detection
- FaceNet embeddings
- Facial similarity comparison
- DeepFace-based emotion analysis
- Camera-based image input

### 🔹 Pattern Finder

Used for:

- Template image selection
- Template matching
- Matching score calculation
- Visualizing the detected region

### 🔹 Video Lab

Used for:

- Uploading video files
- Sampling video frames
- Performing face detection on sampled frames
- Calculating frame-level face statistics

### 🔹 Reports

Provides a session-level summary of operations performed within the application and allows the generated information to be exported as CSV.

---

## 🏗️ Project Architecture

The project follows a modular architecture:

```text
iva-assignment-2/
│
├── app.py
│
├── README.md
│
├── requirements.txt
│
├── requirements-deepface.txt
│
└── modules/
    ├── __init__.py
    ├── image_tools.py
    └── face_tools.py
