import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time

from modules.image_tools import (
    decode_upload, image_stats, make_preview,
    preprocess_image, template_search, detect_faces
)
from modules.face_tools import (
    get_facenet_embedding, compare_embeddings, deepface_analyze
)

st.set_page_config(
    page_title="VisionScope",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {font-size: 42px; font-weight: 800; margin-bottom: 0;}
.subtitle {color:#718096; font-size:17px; margin-top:4px;}
.panel {padding:18px; border:1px solid rgba(128,128,128,.20);
        border-radius:16px; margin-bottom:16px;}
.small {color:#718096; font-size:13px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔭 VisionScope</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A modular image and video analytics workspace</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Workspace")
    page = st.radio(
        "Select module",
        ["Overview", "Image Inspector", "Face Lab", "Pattern Finder",
         "Video Lab", "Reports"]
    )
    st.divider()
    st.caption("Python • Streamlit • OpenCV")
    st.caption("Deep-learning modules load only when requested.")

if "report" not in st.session_state:
    st.session_state.report = []

def add_report(module, result, elapsed):
    st.session_state.report.append({
        "Module": module,
        "Result": result,
        "Time (s)": round(elapsed, 4)
    })

if page == "Overview":
    st.subheader("Project modules")
    cols = st.columns(3)
    cards = [
        ("🖼️ Image Inspector", "Image dimensions, brightness, blur and edge analysis."),
        ("👤 Face Lab", "Viola-Jones detection, FaceNet embeddings and optional DeepFace."),
        ("🎯 Pattern Finder", "OpenCV template matching with a visual bounding box."),
        ("🎥 Video Lab", "Frame sampling and face-count analytics."),
        ("📑 Reports", "Review module results and download a CSV report."),
        ("🚀 Deployment", "Designed so optional deep-learning features fail gracefully.")
    ]
    for col, (title, desc) in zip(cols * 2, cards):
        with col:
            st.markdown(f"### {title}")
            st.write(desc)

    st.info(
        "Recommended workflow: start with Image Inspector and Face Lab. "
        "DeepFace is intentionally optional because its model dependencies are heavier."
    )

elif page == "Image Inspector":
    st.header("🖼️ Image Inspector")
    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], key="inspect_upload"
    )
    if uploaded:
        image = decode_upload(uploaded)
        if image is None:
            st.error("The image could not be decoded.")
            st.stop()

        stats = image_stats(image)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Width", f"{stats['width']} px")
        c2.metric("Height", f"{stats['height']} px")
        c3.metric("Brightness", f"{stats['brightness']:.1f}")
        c4.metric("Sharpness", f"{stats['sharpness']:.1f}")

        st.image(make_preview(image), caption="Input image", use_container_width=True)

        if st.button("Run image analysis", type="primary"):
            start = time.perf_counter()
            gray, enhanced, edges = preprocess_image(image)
            elapsed = time.perf_counter() - start

            a, b, c = st.columns(3)
            a.image(gray, caption="Grayscale", use_container_width=True)
            b.image(enhanced, caption="Contrast enhanced", use_container_width=True)
            c.image(edges, caption="Canny edges", use_container_width=True)

            st.success(f"Analysis completed in {elapsed:.3f} seconds.")
            add_report("Image Inspector", f"{stats['width']}x{stats['height']}", elapsed)

elif page == "Face Lab":
    st.header("👤 Face Lab")
    uploaded = st.file_uploader(
        "Upload a face image", type=["jpg", "jpeg", "png"], key="face_upload"
    )
    camera = st.camera_input("Or capture a face with your camera", key="face_camera")

    source = camera if camera is not None else uploaded

    if source:
        image = decode_upload(source)
        if image is None:
            st.error("Unable to read the selected image.")
            st.stop()

        st.image(make_preview(image), caption="Face Lab input", use_container_width=True)

        left, right = st.columns(2)
        with left:
            if st.button("Detect faces", type="primary"):
                start = time.perf_counter()
                boxes = detect_faces(image)
                elapsed = time.perf_counter() - start

                marked = image.copy()
                for idx, (x, y, w, h) in enumerate(boxes, 1):
                    cv2.rectangle(marked, (x, y), (x+w, y+h), (30, 180, 255), 2)
                    cv2.putText(
                        marked, f"Face {idx}", (x, max(20, y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (30, 180, 255), 2
                    )

                st.image(
                    cv2.cvtColor(marked, cv2.COLOR_BGR2RGB),
                    caption=f"{len(boxes)} face(s) detected",
                    use_container_width=True
                )
                st.metric("Detection time", f"{elapsed:.3f}s")
                add_report("Viola-Jones", f"{len(boxes)} face(s)", elapsed)

        with right:
            st.markdown("### FaceNet")
            st.caption(
                "FaceNet is loaded only when this button is pressed. "
                "The embedding is a numerical representation, not a name."
            )
            if st.button("Generate FaceNet embedding"):
                start = time.perf_counter()
                embedding, message = get_facenet_embedding(image)
                elapsed = time.perf_counter() - start

                if embedding is None:
                    st.warning(message)
                else:
                    st.success("Embedding generated.")
                    st.metric("Vector length", len(embedding))
                    st.code(np.round(embedding[:12], 4).tolist())
                    add_report("FaceNet", f"{len(embedding)} dimensions", elapsed)

        st.divider()
        st.subheader("Optional DeepFace analysis")
        st.caption(
            "This module is deliberately separated from the basic face detector. "
            "If DeepFace/TensorFlow is unavailable, the rest of VisionScope continues working."
        )

        if st.button("Run DeepFace"):
            start = time.perf_counter()
            result, message = deepface_analyze(image)
            elapsed = time.perf_counter() - start

            if result is None:
                st.error(message)
                st.info(
                    "Install the optional dependencies from requirements-deepface.txt "
                    "and restart Streamlit."
                )
            else:
                st.json(result)
                add_report("DeepFace", "Analysis completed", elapsed)

        st.divider()
        st.subheader("Compare two faces")
        reference = st.file_uploader(
            "Reference image", type=["jpg", "jpeg", "png"], key="reference_face"
        )

        if reference and st.button("Compare with FaceNet"):
            ref_image = decode_upload(reference)
            if ref_image is None:
                st.error("Could not read the reference image.")
            else:
                with st.spinner("Creating embeddings..."):
                    emb_a, msg_a = get_facenet_embedding(image)
                    emb_b, msg_b = get_facenet_embedding(ref_image)

                if emb_a is None or emb_b is None:
                    st.warning(msg_a if emb_a is None else msg_b)
                else:
                    score = compare_embeddings(emb_a, emb_b)
                    st.metric("Cosine similarity", f"{score:.4f}")
                    st.write(
                        "This score measures embedding similarity; it should not be "
                        "presented as a guaranteed identity decision."
                    )

elif page == "Pattern Finder":
    st.header("🎯 Pattern Finder")
    source_file = st.file_uploader(
        "Main image", type=["jpg", "jpeg", "png"], key="pattern_source"
    )
    template_file = st.file_uploader(
        "Template image", type=["jpg", "jpeg", "png"], key="pattern_template"
    )
    threshold = st.slider("Detection threshold", 0.40, 0.99, 0.75, 0.01)

    if source_file and template_file:
        source = decode_upload(source_file)
        template = decode_upload(template_file)

        if source is None or template is None:
            st.error("One of the images could not be decoded.")
        elif st.button("Search for pattern", type="primary"):
            start = time.perf_counter()
            result, score, location, message = template_search(
                source, template
            )
            elapsed = time.perf_counter() - start

            if result is None:
                st.error(message)
            else:
                st.image(
                    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                    caption="Template search result",
                    use_container_width=True
                )
                st.metric("Best match", f"{score:.2%}")
                st.write(f"Location: {location}")

                if score >= threshold:
                    st.success("Pattern passes the selected threshold.")
                else:
                    st.warning("Best match is below the selected threshold.")

                add_report("Template Matching", f"{score:.2%}", elapsed)

elif page == "Video Lab":
    st.header("🎥 Video Lab")
    st.caption(
        "This lightweight module samples frames instead of running an expensive "
        "deep model on every frame."
    )
    video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"], key="video_upload"
    )
    sample_every = st.slider("Analyze every Nth frame", 1, 30, 10)

    if video:
        data = video.read()
        if st.button("Analyze video", type="primary"):
            import tempfile, os

            suffix = os.path.splitext(video.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                path = tmp.name

            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frame_no = 0
            samples = 0
            face_counts = []

            start = time.perf_counter()
            progress = st.progress(0)

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_no % sample_every == 0:
                    boxes = detect_faces(frame)
                    face_counts.append(len(boxes))
                    samples += 1

                frame_no += 1
                if total:
                    progress.progress(min(frame_no / total, 1.0))

            cap.release()
            os.unlink(path)

            elapsed = time.perf_counter() - start
            avg_faces = float(np.mean(face_counts)) if face_counts else 0.0
            max_faces = max(face_counts) if face_counts else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Frames", frame_no)
            c2.metric("Samples", samples)
            c3.metric("Avg. faces", f"{avg_faces:.2f}")
            c4.metric("Peak faces", max_faces)

            if fps:
                st.write(f"Video FPS: {fps:.2f}")
            st.success(f"Video analysis completed in {elapsed:.2f} seconds.")
            add_report("Video Lab", f"Peak faces: {max_faces}", elapsed)

elif page == "Reports":
    st.header("📑 Session Report")

    if not st.session_state.report:
        st.info("Run one or more modules to populate this report.")
    else:
        df = pd.DataFrame(st.session_state.report)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            file_name="visionscope_report.csv",
            mime="text/csv"
        )
        if st.button("Clear session report"):
            st.session_state.report = []
            st.rerun()
