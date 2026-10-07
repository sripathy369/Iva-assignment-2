import os
import tempfile
import numpy as np
import cv2

def _save_temp(image):
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(path, image)
    return path

def get_facenet_embedding(image):
    try:
        from deepface import DeepFace
    except Exception as exc:
        return None, (
            "FaceNet is unavailable because DeepFace/TensorFlow is not installed "
            f"correctly. Details: {exc}"
        )

    path = _save_temp(image)
    try:
        result = DeepFace.represent(
            img_path=path,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=True
        )
        if not result:
            return None, "No usable face embedding was returned."

        vector = np.asarray(result[0]["embedding"], dtype=np.float32)
        return vector, "OK"
    except Exception as exc:
        return None, f"FaceNet processing failed: {exc}"
    finally:
        if os.path.exists(path):
            os.remove(path)

def compare_embeddings(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

def deepface_analyze(image):
    try:
        from deepface import DeepFace
    except Exception as exc:
        return None, f"DeepFace is not available: {exc}"

    path = _save_temp(image)
    try:
        result = DeepFace.analyze(
            img_path=path,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=True
        )

        if isinstance(result, list):
            result = result[0]

        # Keep the report focused on the model output we actually need.
        return {
            "dominant_emotion": result.get("dominant_emotion"),
            "emotion_scores": result.get("emotion")
        }, "OK"

    except Exception as exc:
        return None, f"DeepFace analysis failed: {exc}"
    finally:
        if os.path.exists(path):
            os.remove(path)
