# app.py
import streamlit as st
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import Callback

# -------------------------
# Config & constants
# -------------------------
st.set_page_config(page_title="Teachable Machine Clone (Pro)", layout="wide", page_icon="🤖")
BASE_DIR = os.path.abspath(".")
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
DEFAULT_EPOCHS = 10

os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------
# Session state init
# -------------------------
if 'classes' not in st.session_state:
    st.session_state['classes'] = []
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
if 'train_gen' not in st.session_state:
    st.session_state['train_gen'] = None
if 'val_gen' not in st.session_state:
    st.session_state['val_gen'] = None
if 'history_df' not in st.session_state:
    st.session_state['history_df'] = pd.DataFrame(columns=['epoch','accuracy','val_accuracy','loss','val_loss'])
if 'history_obj' not in st.session_state:
    st.session_state['history_obj'] = None
if 'model_obj' not in st.session_state:
    st.session_state['model_obj'] = None

# -------------------------
# Helper functions
# -------------------------
def save_labels(labels):
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def build_cnn(num_classes, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def predict_topk(model, img_array, classes, k=3):
    probs = model.predict(img_array)[0]
    idxs = probs.argsort()[-k:][::-1]
    return [(classes[i], float(probs[i])) for i in idxs]

# -------------------------
# UI layout
# -------------------------
st.title("🎓 Teachable Machine Clone — Pro")
st.markdown(
    "Create classes, upload images, preview augmentation, train a model live, evaluate, predict, visualize and download reports."
)

col_left, col_right = st.columns([1.2, 2])

# -------------------------
# Left: Controls (classes, upload, augmentation settings)
# -------------------------
with col_left:
    st.header("Setup & Data")

    # Add class
    st.subheader("1) Manage classes")
    new_class = st.text_input("Add class (type name and press Add)", key="txt_new_class")
    if st.button("Add Class"):
        if new_class.strip():
            clsname = new_class.strip()
            cls_folder = os.path.join(DATA_DIR, clsname)
            os.makedirs(cls_folder, exist_ok=True)
            if clsname not in st.session_state['classes']:
                st.session_state['classes'].append(clsname)
            st.success(f"Class '{clsname}' added (folder created).")
        else:
            st.warning("Enter a non-empty class name.")

    if st.session_state['classes']:
        st.info(f"Classes (UI list): {', '.join(st.session_state['classes'])}")
    else:
        st.info("No classes added yet.")

    st.markdown("---")

    # Upload images per class
    st.subheader("2) Upload images")
    if st.session_state['classes']:
        for cls in st.session_state['classes']:
            with st.expander(f"Upload images for '{cls}'", expanded=False):
                files = st.file_uploader(f"Pick images for {cls}", accept_multiple_files=True, type=["png","jpg","jpeg"], key=f"upload_{cls}")
                if files:
                    folder = os.path.join(DATA_DIR, cls)
                    os.makedirs(folder, exist_ok=True)
                    count = 0
                    for f in files:
                        out_path = os.path.join(folder, f.name)
                        if os.path.exists(out_path):
                            base, ext = os.path.splitext(f.name)
                            i = 1
                            while os.path.exists(os.path.join(folder, f"{base}_{i}{ext}")):
                                i += 1
                            out_path = os.path.join(folder, f"{base}_{i}{ext}")
                        with open(out_path, "wb") as out:
                            out.write(f.getbuffer())
                        count += 1
                    st.success(f"Saved {count} images to {folder}")
    else:
        st.info("Add classes first (left) to upload images.")

    st.markdown("---")

    # Data augmentation preview
    st.subheader("3) Augmentation preview")
    with st.form("augment_form"):
        rot = st.slider("Rotation range (degrees)", 0, 90, 20)
        zoom = st.slider("Zoom range", 0.0, 0.6, 0.15)
        hflip = st.checkbox("Horizontal flip")
        sample_class = st.selectbox("Preview images from class", options=(st.session_state['classes'] if st.session_state['classes'] else ["None"]))
        submitted = st.form_submit_button("Preview Augmentation")
        if submitted:
            if sample_class == "None":
                st.warning("No classes available to preview.")
            else:
                sample_folder = os.path.join(DATA_DIR, sample_class)
                files = [f for f in os.listdir(sample_folder) if os.path.isfile(os.path.join(sample_folder, f))]
                if not files:
                    st.warning("No images found in that class folder.")
                else:
                    first = os.path.join(sample_folder, files[0])
                    img = load_img(first, target_size=IMG_SIZE)
                    x = img_to_array(img) / 255.0
                    x = x.reshape((1,) + x.shape)
                    datagen = ImageDataGenerator(rotation_range=rot, zoom_range=zoom, horizontal_flip=hflip)
                    aug = datagen.flow(x, batch_size=1)
                    cols = st.columns(5)
                    for i in range(5):
                        batch = next(aug)[0]
                        cols[i].image(batch, use_column_width=True)

    st.markdown("---")

    # Class counts and balance check
    st.subheader("4) Class balance check")
    if st.button("Scan dataset folders"):
        folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        counts = {}
        for f in folders:
            files = [x for x in os.listdir(os.path.join(DATA_DIR, f)) if os.path.isfile(os.path.join(DATA_DIR, f, x))]
            counts[f] = len(files)
        if counts:
            st.table(pd.DataFrame.from_dict(counts, orient='index', columns=['image_count']))
            mins = min(counts.values())
            maxs = max(counts.values())
            if maxs > mins * 5:
                st.warning("High class imbalance detected (largest class >5x smallest). Consider adding images for smaller classes.")

            # Class distribution chart
            st.subheader("Class Distribution Chart")
            st.bar_chart(pd.DataFrame.from_dict(counts, orient='index', columns=['Images']))
        else:
            st.info("No class folders with images found.")

# -------------------------
# Right: Training, evaluation & prediction
# -------------------------
with col_right:
    st.header("Train & Evaluate")

    # Training controls
    st.subheader("5) Train model")
    epochs = st.number_input("Epochs", min_value=1, max_value=200, value=DEFAULT_EPOCHS)
    if st.button("Start Training"):
        folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        valid_folders = []
        for f in folders:
            files = [x for x in os.listdir(os.path.join(DATA_DIR, f)) if os.path.isfile(os.path.join(DATA_DIR, f, x))]
            if len(files) > 0:
                valid_folders.append(f)
        if len(valid_folders) < 2:
            st.error("Need at least 2 classes with images to train.")
        else:
            datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
            train_gen = datagen.flow_from_directory(
                DATA_DIR,
                target_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )
            val_gen = datagen.flow_from_directory(
                DATA_DIR,
                target_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )

            detected_classes = list(train_gen.class_indices.keys())
            st.session_state['classes'] = detected_classes
            save_labels(detected_classes)
            num_classes = train_gen.num_classes
            st.write(f"Detected {num_classes} classes: {detected_classes}")

            model = build_cnn(num_classes)
            st.session_state['model_obj'] = model
            st.session_state['train_gen'] = train_gen
            st.session_state['val_gen'] = val_gen

            # prepare charts
            st.subheader("Training progress (live)")
            acc_chart = st.empty()
            loss_chart = st.empty()
            progress_bar = st.progress(0)

            class LivePlotCallback(Callback):
                def __init__(self, epochs):
                    super().__init__()
                    self.epochs = epochs
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    new_row = pd.DataFrame([{
                        'epoch': epoch+1,
                        'accuracy': logs.get('accuracy'),
                        'val_accuracy': logs.get('val_accuracy'),
                        'loss': logs.get('loss'),
                        'val_loss': logs.get('val_loss')
                    }])
                    st.session_state['history_df'] = pd.concat([st.session_state['history_df'], new_row], ignore_index=True)
                    acc_chart.line_chart(st.session_state['history_df'][['accuracy','val_accuracy']])
                    loss_chart.line_chart(st.session_state['history_df'][['loss','val_loss']])
                    progress = int(((epoch+1)/self.epochs) * 100)
                    progress_bar.progress(progress)

            st.info("Training started — this may take a while depending on your machine.")
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=[LivePlotCallback(epochs)]
            )

            model.save(MODEL_PATH)
            st.session_state['model_trained'] = True
            st.session_state['history_obj'] = history
            st.session_state['model_obj'] = model
            st.success(f"Training finished. Model saved to {MODEL_PATH}")
            st.balloons()

    st.markdown("---")

    # Performance dashboard
    st.subheader("6) Performance Dashboard")
    if st.session_state['model_trained'] and st.session_state['history_obj'] is not None:
        hist = st.session_state['history_obj'].history
        fig, axes = plt.subplots(1,2, figsize=(12,4))
        axes[0].plot(hist['accuracy'], label='Train Acc')
        axes[0].plot(hist['val_accuracy'], label='Val Acc')
        axes[0].set_title("Accuracy")
        axes[0].legend()
        axes[1].plot(hist['loss'], label='Train Loss')
        axes[1].plot(hist['val_loss'], label='Val Loss')
        axes[1].set_title("Loss")
        axes[1].legend()
        st.pyplot(fig)

        st.markdown("**Confusion Matrix & Classification Report (on validation set)**")
        val_gen = st.session_state['val_gen']
        model = st.session_state['model_obj']
        val_steps = int(np.ceil(val_gen.samples / val_gen.batch_size))
        preds = model.predict(val_gen, steps=val_steps, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        true_labels = val_gen.classes

        cm = confusion_matrix(true_labels, pred_labels)
        fig2, ax2 = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
                    xticklabels=st.session_state['classes'], yticklabels=st.session_state['classes'])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig2)

        try:
            report = classification_report(true_labels, pred_labels, target_names=st.session_state['classes'], output_dict=True)
            st.table(pd.DataFrame(report).transpose())
        except ValueError:
            st.warning("Mismatch between number of classes and predictions. Check your dataset.")

    else:
        st.info("No trained model yet. Train a model to see performance dashboard.")

    st.markdown("---")

    # Prediction UI
    st.subheader("7) Predict / Test")
    tab_upload, tab_webcam = st.tabs(["Upload image", "Webcam capture"])

    with tab_upload:
        uploaded = st.file_uploader("Upload image for prediction", accept_multiple_files=False, type=["png","jpg","jpeg"])
        if uploaded:
            if not st.session_state['model_trained']:
                st.warning("Train a model first.")
            else:
                model = load_model(MODEL_PATH)
                img = load_img(uploaded, target_size=IMG_SIZE)
                arr = img_to_array(img)/255.0
                arr = np.expand_dims(arr, axis=0)
                top3 = predict_topk(model, arr, st.session_state['classes'], k=3)
                st.image(img, caption="Input image", use_column_width=False)
                
                st.markdown("**Top-3 predictions with confidence:**")
                for name, conf in top3:
                    st.write(f"{name}: {conf*100:.2f}%")
                    st.progress(int(conf*100))

                # Export prediction to CSV
                pred_df = pd.DataFrame([{
                    "image": uploaded.name,
                    "pred_class": top3[0][0],
                    "confidence": top3[0][1]
                }])
                csv_file = "prediction_report.csv"
                pred_df.to_csv(csv_file, index=False)
                st.download_button("Download Prediction CSV", data=open(csv_file,"rb"), file_name="prediction_report.csv")

    with tab_webcam:
        st.write("Capture from webcam (single snapshot, then predict)")
        cap_img = st.camera_input("Capture image")
        if cap_img:
            if not st.session_state['model_trained']:
                st.warning("Train a model first.")
            else:
                model = load_model(MODEL_PATH)
                img = Image.open(cap_img).resize(IMG_SIZE)
                arr = img_to_array(img)/255.0
                arr = np.expand_dims(arr, axis=0)
                top3 = predict_topk(model, arr, st.session_state['classes'], k=3)
                st.image(img, caption="Captured image", use_column_width=False)
                
                st.markdown("**Top-3 predictions with confidence:**")
                for name, conf in top3:
                    st.write(f"{name}: {conf*100:.2f}%")
                    st.progress(int(conf*100))

                pred_df = pd.DataFrame([{
                    "image": "Webcam Capture",
                    "pred_class": top3[0][0],
                    "confidence": top3[0][1]
                }])
                csv_file = "prediction_report.csv"
                pred_df.to_csv(csv_file, index=False)
                st.download_button("Download Prediction CSV", data=open(csv_file,"rb"), file_name="prediction_report.csv")

    st.markdown("---")

    # Model download
    st.subheader("8) Export / Download")
    if st.session_state['model_trained'] and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            st.download_button("Download trained model (.keras)", f, file_name="trained_model.keras")
        save_labels(st.session_state['classes'])
        with open(LABELS_PATH, "r", encoding="utf-8") as lf:
            st.download_button("Download labels (JSON)", lf, file_name="labels.json")
    else:
        st.info("Train a model to enable exports.")

# Footer
st.markdown("---")
st.caption("Tips: 1) Ensure each class folder has enough images. 2) Use augmentation preview. 3) Rescan folders after adding classes outside the UI.")
