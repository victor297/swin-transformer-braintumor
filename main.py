import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the trained model
model = tf.keras.models.load_model('brain_tumor_detection_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))
    image = image / 255.0  # normalize the image
    image = image.reshape((1, 64, 64, 3))  # ensure the image has 3 channels
    return image

# Function to predict an image
def predict_image(img_array):
    prediction = model.predict(img_array)
    return 'Yes' if prediction[0][0] > 0.5 else 'No'

# Function to visualize metrics
def visualize_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=['No', 'Yes'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

    report = classification_report(y_true, y_pred, target_names=['No', 'Yes'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

# Streamlit app
st.title('Brain Tumor Detection')
st.write('By Olukotun Stephen timileyin 20/47cs/01167')
st.write('Upload an MRI image to detect brain tumor.')

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

if uploaded_file is not None:
    # Read the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, channels="BGR")

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    prediction = predict_image(preprocessed_image)

    # Display the result
    if prediction == 'Yes':
        st.write('The model predicts: **Brain Tumor Detected**')
    else:
        st.write('The model predicts: **No Brain Tumor Detected**')

# Directory paths for testing
no_dir = './dataset/no'
yes_dir = './dataset/yes'

# Button to generate metrics
if st.button('Generate Metrics'):
    y_true = []
    y_pred = []

    # Predict on all images in the no_dir
    for img_name in os.listdir(no_dir):
        img_path = os.path.join(no_dir, img_name)
        img_array = preprocess_image(cv2.imread(img_path))
        prediction = predict_image(img_array)
        y_pred.append(prediction)
        y_true.append('No')

    # Predict on all images in the yes_dir
    for img_name in os.listdir(yes_dir):
        img_path = os.path.join(yes_dir, img_name)
        img_array = preprocess_image(cv2.imread(img_path))
        prediction = predict_image(img_array)
        y_pred.append(prediction)
        y_true.append('Yes')

    # Visualize the metrics
    visualize_metrics(y_true, y_pred)
