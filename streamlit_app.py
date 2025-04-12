import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import gdown

st.set_page_config(page_title="Plant Disease Detector", layout="wide")
st.title("🌿 Plant Disease Detection with Xception")

image_size = (224, 224)
batch_size = 32


@st.cache_data
def load_data(dataset_path):
    images, labels = [], []
    for subfolder in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, subfolder)
        for image_filename in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_filename)
            images.append(image_path)
            labels.append(subfolder)
    return pd.DataFrame({'image': images, 'label': labels})


# def load_model():
#  model_path = os.path.join(os.path.dirname(
#     __file__), "xception_model.keras")
# return keras.models.load_model(model_path)


MODEL_FILE_ID = "1gvre33K5x6ZYrUgGdhk5l6FqNMXTCJi2"
MODEL_FILENAME = "xception_model.keras"


def download_model_from_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False, fuzzy=True)


@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 10000:
        # If file exists but is invalid, remove it
        if os.path.exists(model_path):
            os.remove(model_path)
        with st.spinner("Downloading model from Google Drive..."):
            download_model_from_drive(MODEL_FILE_ID, model_path)

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 10000:
        raise RuntimeError("❌ Downloaded model is invalid or incomplete.")

    return keras.models.load_model(model_path)


# Sidebar
with st.sidebar:
    st.header("Configuration")
    mode = st.radio("Mode", ["Predict with existing model", "Train new model"])

class_names = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", "Background_without_leaves", "Blueberry___healthy",
               "Cherry___Powdery_mildew", "Cherry___healthy", "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
               "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy", "Orange___Haunglongbing_",
               "Peach___Bacterial_spot", "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
               "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
               "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
               "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"]

if mode == "Predict with existing model":
    st.subheader("🔍 Upload an Image for Prediction")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, image_size)
        img_array = img_resized / 255.0
        img_expanded = np.expand_dims(img_array, axis=0)

        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

        model = load_model()
        prediction = model.predict(img_expanded)
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]

        # class_names = list(model.output_names)  # fallback, or manually define
        st.success(
            f"**Prediction:** {class_names[class_idx]} with {confidence:.2%} confidence")
    # else:
       # st.info("Upload an image to start.")

elif mode == "Train new model":
    dataset_path = st.text_input("Enter dataset folder path")
    show_preview = st.checkbox("Show 50 random images")
    train_button = st.button("Start Training")

    if dataset_path:
        df = load_data(dataset_path)
        st.success(
            f"Loaded {len(df)} images from {len(df.label.unique())} classes.")
        st.subheader("Class Distribution")
        st.bar_chart(df['label'].value_counts())

        if show_preview:
            st.subheader("📸 Random Sample Images")
            sample_df = df.sample(50)
            cols = st.columns(5)
            for i, row in sample_df.iterrows():
                img = cv2.imread(row['image'])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cols[i % 5].image(img, caption=row['label'],
                                  use_column_width=True)

        # split
        X_train, X_test1, y_train, y_test1 = train_test_split(
            df['image'], df['label'], test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
            X_test1, y_test1, test_size=0.5, random_state=42)

        df_train = pd.DataFrame({'image': X_train, 'label': y_train})
        df_val = pd.DataFrame({'image': X_val, 'label': y_val})
        df_test = pd.DataFrame({'image': X_test, 'label': y_test})

        datagen = ImageDataGenerator(rescale=1./255)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        train_generator = train_datagen.flow_from_dataframe(
            df_train, x_col='image', y_col='label', target_size=image_size,
            batch_size=batch_size, shuffle=True)
        val_generator = datagen.flow_from_dataframe(
            df_val, x_col='image', y_col='label', target_size=image_size,
            batch_size=batch_size, shuffle=False)
        test_generator = datagen.flow_from_dataframe(
            df_test, x_col='image', y_col='label', target_size=image_size,
            batch_size=batch_size, shuffle=False)

        class_names = list(train_generator.class_indices.keys())

        if train_button:
            base_model = tf.keras.applications.Xception(
                weights='imagenet', include_top=False,
                input_shape=(224, 224, 3), pooling='max')

            model = keras.models.Sequential([
                base_model,
                keras.layers.BatchNormalization(),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(len(class_names), activation='softmax')
            ])

            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy', keras.metrics.AUC()])

            history = model.fit(train_generator, epochs=5,
                                validation_data=val_generator)

            st.success("Model training completed. Saving model...")
            model.save("xception_model.keras")
            st.info("Model saved as `xception_model.keras`.")
