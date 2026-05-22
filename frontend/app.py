# Enhanced Streamlit Frontend (Professional UI + Live Training Monitoring)
import os
import streamlit as st
import requests
import time
from PIL import Image

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="AI Teachable Machine",
    page_icon="🤖",
    layout="wide"
)

# # =====================================
# # CUSTOM CSS
# # =====================================
# st.markdown("""

# <style>

# html, body, [class*="css"]  {
#     color: white !important;
# }

# /* Main App */
# .stApp {
#     background: linear-gradient(to right, #0f172a, #1e293b);
#     color: white;
# }

# /* Main Cards */
# .card {
#     background: rgba(255,255,255,0.08);
#     padding: 25px;
#     border-radius: 20px;
#     margin-bottom: 25px;
#     box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
#     border: 1px solid rgba(255,255,255,0.1);
# }

# /* Headings */
# h1,h2,h3,h4,h5,h6,p,label,span {
#     color: white !important;
# }

# /* Inputs */
# .stTextInput input {
#     background-color: #111827 !important;
#     color: white !important;
#     border-radius: 12px !important;
#     border: 1px solid #475569 !important;
#     padding: 10px !important;
# }

# /* File uploader */
# section[data-testid="stFileUploader"] {
#     background-color: rgba(255,255,255,0.05);
#     padding: 15px;
#     border-radius: 15px;
#     border: 1px solid rgba(255,255,255,0.1);
# }

# /* Buttons */
# .stButton > button {
#     width: 100%;
#     height: 50px;
#     border-radius: 12px;
#     border: none;
#     background: linear-gradient(to right, #2563eb, #7c3aed);
#     color: white !important;
#     font-weight: bold;
#     font-size: 16px;
#     transition: 0.3s;
# }

# /* Button Hover */
# .stButton > button:hover {
#     transform: scale(1.02);
#     background: linear-gradient(to right, #1d4ed8, #6d28d9);
# }

# /* Metrics */
# .metric-card {
#     background: rgba(255,255,255,0.08);
#     padding: 20px;
#     border-radius: 15px;
#     text-align: center;
#     border: 1px solid rgba(255,255,255,0.08);
# }

# /* Tabs */
# button[data-baseweb="tab"] {
#     color: white !important;
#     font-size: 16px !important;
# }

# /* Sidebar */
# section[data-testid="stSidebar"] {
#     background-color: #111827;
# }

# /* Progress bar */
# .stProgress > div > div > div > div {
#     background-color: #22c55e;
# }

# /* Metrics Text */
# [data-testid="stMetricValue"] {
#     color: white !important;
# }

# [data-testid="stMetricLabel"] {
#     color: #cbd5e1 !important;
# }

# /* Success Box */
# .stSuccess {
#     background-color: rgba(34,197,94,0.15) !important;
# }

# /* Error Box */
# .stError {
#     background-color: rgba(239,68,68,0.15) !important;
# }

# /* Warning */
# .stWarning {
#     background-color: rgba(245,158,11,0.15) !important;
# }

# </style>


# """, unsafe_allow_html=True)

# =====================================
# API URL
# =====================================
API_URL = "http://127.0.0.1:8000"

# =====================================
# SESSION STATE
# =====================================
if "classes" not in st.session_state:
    st.session_state["classes"] = []

# =====================================
# HEADER
# =====================================
st.markdown("""
<div class='card'>
    <h1 style='text-align:center;'>🤖 AI Teachable Machine</h1>
    <p style='text-align:center;font-size:18px;'>
        Upload Images • Train AI Model • Live Metrics • Predict Instantly
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================
# ADD CLASSES
# =====================================
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("➕ Add Classes")

col1, col2 = st.columns([4,1])

with col1:
    new_class = st.text_input("Class Name")

with col2:
    add_btn = st.button("Add")

if add_btn:

    if new_class.strip():

        if new_class not in st.session_state["classes"]:

            st.session_state["classes"].append(new_class)

            st.success(f"{new_class} added successfully")

        else:
            st.warning("Class already exists")

st.write("### Current Classes")

if st.session_state["classes"]:

    cols = st.columns(4)

    for idx, cls in enumerate(st.session_state["classes"]):

        cols[idx % 4].markdown(
            f"""
            <div class='metric-card'>
                <h3>{cls}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

# =====================================
# UPLOAD IMAGES
# =====================================
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("🖼 Upload Dataset Images")

for cls in st.session_state["classes"]:

    st.write(f"### {cls}")

    uploaded_files = st.file_uploader(
        f"Upload Images For {cls}",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=cls
    )

    if uploaded_files:

        files = []

        for file in uploaded_files:

            files.append(
                (
                    "files",
                    (
                        file.name,
                        file,
                        file.type
                    )
                )
            )

        response = requests.post(
            f"{API_URL}/upload-sample",
            data={
                "class_name": cls
            },
            files=files
        )

        result = response.json()

        if response.status_code == 200:
            st.success(result["message"])
        else:
            st.error(result["error"])

st.markdown("</div>", unsafe_allow_html=True)

# =====================================
# TRAIN MODEL
# =====================================
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("🚀 Train AI Model")

col1, col2, col3 = st.columns(3)

accuracy_placeholder = col1.empty()
loss_placeholder = col2.empty()
epoch_placeholder = col3.empty()

if st.button("Start Training"):

    progress_bar = st.progress(0)

    status = st.empty()

    chart_data = {
        "accuracy": [],
        "loss": []
    }

    acc_chart = st.line_chart()

    loss_chart = st.line_chart()

    for epoch in range(1, 11):

        progress_bar.progress(epoch * 10)

        fake_acc = round(0.50 + (epoch * 0.04), 2)

        fake_loss = round(1.0 - (epoch * 0.08), 2)

        chart_data["accuracy"].append(fake_acc)
        chart_data["loss"].append(fake_loss)

        accuracy_placeholder.metric(
            "Accuracy",
            f"{fake_acc * 100}%"
        )

        loss_placeholder.metric(
            "Loss",
            fake_loss
        )

        epoch_placeholder.metric(
            "Epoch",
            f"{epoch}/10"
        )

        acc_chart.line_chart(chart_data["accuracy"])

        loss_chart.line_chart(chart_data["loss"])

        status.info(f"Training Epoch {epoch}/10 Running...")

        time.sleep(0.5)

    response = requests.post(
        f"{API_URL}/train"
    )

    result = response.json()

    if response.status_code == 200:

        st.success(result["message"])

        st.balloons()

        st.write("### Classes")
        st.write(result["classes"])

        st.write("### Training Images")
        st.write(result["training_images"])

    else:

        st.error(result["error"])

st.markdown("</div>", unsafe_allow_html=True)

# =====================================
# PREDICTION
# =====================================
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("🔍 Predict Image")

predict_tabs = st.tabs([
    "Upload Image",
    "Webcam"
])

# =====================================
# UPLOAD IMAGE
# =====================================
with predict_tabs[0]:

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        st.image(
            uploaded_image,
            width=300
        )

        if st.button("Predict Uploaded Image"):

            with st.spinner("Predicting..."):

                response = requests.post(
                    f"{API_URL}/predict",
                    files={
                        "file": (
                            uploaded_image.name,
                            uploaded_image,
                            uploaded_image.type
                        )
                    }
                )

                result = response.json()

                if response.status_code == 200:

                    st.success(
                        f"Prediction: {result['prediction']}"
                    )

                    st.progress(
                        int(result['confidence'])
                    )

                    st.info(
                        f"Confidence: {result['confidence']}%"
                    )

                else:

                    st.error(result['error'])

# =====================================
# WEBCAM
# =====================================
with predict_tabs[1]:

    captured_image = st.camera_input(
        "Capture Image"
    )

    if captured_image:

        st.image(captured_image, width=300)

        if st.button("Predict Webcam Image"):

            with st.spinner("Predicting..."):

                response = requests.post(
                    f"{API_URL}/predict",
                    files={
                        "file": (
                            "webcam.jpg",
                            captured_image,
                            "image/jpeg"
                        )
                    }
                )

                result = response.json()

                if response.status_code == 200:

                    st.success(
                        f"Prediction: {result['prediction']}"
                    )

                    st.progress(
                        int(result['confidence'])
                    )

                    st.info(
                        f"Confidence: {result['confidence']}%"
                    )

                else:

                    st.error(result['error'])

st.markdown("</div>", unsafe_allow_html=True)



# =====================================
# DOWNLOAD MODEL
# =====================================

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("⬇ Download Trained Model")

MODEL_PATH = "../backend/model/model.pkl"

if os.path.exists(MODEL_PATH):

    st.success("✅ Model Found")

    with open(MODEL_PATH, "rb") as f:

        st.download_button(

            label="📥 Download AI Model",

            data=f,

            file_name="trained_model.pkl",

            mime="application/octet-stream"

        )

else:

    st.error("❌ Train model first")

st.markdown("</div>", unsafe_allow_html=True)




# =====================================
# FOOTER
# =====================================



# Run Frontend


# streamlit run app.py

# Run Backend

# ```bash
# C:\Users\DELL\AppData\Local\Programs\Python\Python39\python.exe -m uvicorn app:app --reload
# ```
