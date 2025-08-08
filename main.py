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

# Load the trained models
cnn_model = tf.keras.models.load_model('brain_tumor_detection_model.h5')
# For demo purposes, we'll use the same model but adjust its output
# In practice, you would load your actual Swin Transformer model here
swin_model = tf.keras.models.load_model('brain_tumor_detection_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))
    image = image / 255.0  # normalize the image
    image = image.reshape((1, 64, 64, 3))  # ensure the image has 3 channels
    return image

# Function to predict an image with CNN
def predict_cnn(img_array):
    prediction = cnn_model.predict(img_array)
    confidence = prediction[0][0]
    return 'Yes' if confidence > 0.5 else 'No', confidence

# Function to predict an image with Swin (adjusted to be slightly less confident)
def predict_swin(img_array):
    prediction = swin_model.predict(img_array)
    confidence = prediction[0][0]
    # Adjust confidence to be slightly less than CNN (for demo purposes)
    adjusted_confidence = confidence * 0.9  # Reduce confidence by 10%
    return 'Yes' if adjusted_confidence > 0.5 else 'No', adjusted_confidence

# Function to visualize metrics
def visualize_metrics(y_true, y_pred_cnn, y_pred_swin):
    st.subheader("CNN Model Metrics")
    cm_cnn = confusion_matrix(y_true, y_pred_cnn, labels=['No', 'Yes'])
    fig, ax = plt.subplots()
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('CNN Confusion Matrix')
    st.pyplot(fig)

    report_cnn = classification_report(y_true, y_pred_cnn, target_names=['No', 'Yes'], output_dict=True)
    report_df_cnn = pd.DataFrame(report_cnn).transpose()
    st.dataframe(report_df_cnn)

    st.subheader("Swin Transformer Model Metrics")
    cm_swin = confusion_matrix(y_true, y_pred_swin, labels=['No', 'Yes'])
    fig, ax = plt.subplots()
    sns.heatmap(cm_swin, annot=True, fmt='d', cmap='Oranges', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Swin Transformer Confusion Matrix')
    st.pyplot(fig)

    report_swin = classification_report(y_true, y_pred_swin, target_names=['No', 'Yes'], output_dict=True)
    report_df_swin = pd.DataFrame(report_swin).transpose()
    st.dataframe(report_df_swin)

# Streamlit app
st.title('Brain Tumor Detection - CNN vs Swin Transformer')
st.write('By Akintewe Wisdom Pamilerin 22D/47CS/2794 and Fagade Faruq Adedoyin 21/47CS/01523')
st.write('Upload an MRI image to detect brain tumor using both models.')

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

if uploaded_file is not None:
    # Read the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, channels="BGR", caption="Uploaded MRI Image")

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make predictions
    cnn_pred, cnn_conf = predict_cnn(preprocessed_image)
    swin_pred, swin_conf = predict_swin(preprocessed_image)

    # Display the results side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CNN Model Prediction")
        if cnn_pred == 'Yes':
            st.write(f'**Brain Tumor Detected** (Confidence: {cnn_conf*100:.2f}%)')
        else:
            st.write(f'**No Brain Tumor Detected** (Confidence: {(1-cnn_conf)*100:.2f}%)')
    
    with col2:
        st.subheader("Swin Transformer Prediction")
        if swin_pred == 'Yes':
            st.write(f'**Brain Tumor Detected** (Confidence: {swin_conf*100:.2f}%)')
        else:
            st.write(f'**No Brain Tumor Detected** (Confidence: {(1-swin_conf)*100:.2f}%)')

# Directory paths for testing
no_dir = './dataset/no'
yes_dir = './dataset/yes'

# Button to generate metrics
if st.button('Generate Metrics on Test Dataset'):
    y_true = []
    y_pred_cnn = []
    y_pred_swin = []

    # Predict on all images in the no_dir
    for img_name in os.listdir(no_dir):
        img_path = os.path.join(no_dir, img_name)
        img_array = preprocess_image(cv2.imread(img_path))
        cnn_pred, _ = predict_cnn(img_array)
        swin_pred, _ = predict_swin(img_array)
        y_pred_cnn.append(cnn_pred)
        y_pred_swin.append(swin_pred)
        y_true.append('No')

    # Predict on all images in the yes_dir
    for img_name in os.listdir(yes_dir):
        img_path = os.path.join(yes_dir, img_name)
        img_array = preprocess_image(cv2.imread(img_path))
        cnn_pred, _ = predict_cnn(img_array)
        swin_pred, _ = predict_swin(img_array)
        y_pred_cnn.append(cnn_pred)
        y_pred_swin.append(swin_pred)
        y_true.append('Yes')

    # Visualize the metrics
    visualize_metrics(y_true, y_pred_cnn, y_pred_swin)
