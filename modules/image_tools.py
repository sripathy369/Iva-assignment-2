import cv2
import numpy as np

def decode_upload(uploaded_file):
    raw = uploaded_file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def make_preview(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def image_stats(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "width": image.shape[1],
        "height": image.shape[0],
        "channels": image.shape[2] if image.ndim == 3 else 1,
        "brightness": float(np.mean(gray)),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var())
    }

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smooth)

    edges = cv2.Canny(enhanced, 50, 150)
    return gray, enhanced, edges

def template_search(source, template):
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    th, tw = template_gray.shape
    sh, sw = source_gray.shape

    if th > sh or tw > sw:
        return None, 0.0, None, "The template must be smaller than the main image."

    scores = cv2.matchTemplate(
        source_gray, template_gray, cv2.TM_CCOEFF_NORMED
    )
    _, best_score, _, best_point = cv2.minMaxLoc(scores)

    output = source.copy()
    x, y = best_point
    cv2.rectangle(output, (x, y), (x + tw, y + th), (40, 210, 120), 3)

    return output, float(best_score), best_point, "OK"

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )
    return faces
