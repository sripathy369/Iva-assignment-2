# VisionScope

A modular Streamlit image and video analytics project.

## Core features

- OpenCV image statistics
- Grayscale / CLAHE / Canny preprocessing
- Viola-Jones face detection
- Template matching
- Camera capture through Streamlit
- Lightweight video frame sampling
- Session CSV report
- Optional FaceNet and DeepFace modules

## Run

Create an environment, then:

    pip install -r requirements.txt
    streamlit run app.py

The basic application does not require TensorFlow or DeepFace.

## Optional face AI

If your machine has enough resources:

    pip install -r requirements-deepface.txt
    streamlit run app.py

FaceNet and DeepFace are loaded only when the corresponding feature is used.

## Project structure

    visionscope/
    ├── app.py
    ├── requirements.txt
    ├── requirements-deepface.txt
    ├── README.md
    └── modules/
        ├── __init__.py
        ├── image_tools.py
        └── face_tools.py

## Important

Model predictions are estimates. Face embeddings should be treated as numerical
features for demonstration and not as guaranteed identity proof.
