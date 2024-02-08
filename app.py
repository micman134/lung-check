import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Load the saved model
try:
    model = tf.keras.models.load_model('cancer_detection_model2.h5')
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")

# Define the relatable class labels
class_labels = ['adenocarcinoma', 'large.cell.carcinoma', 'squamous.cell.carcinoma', 'normal']

# Function to display limited rows of the pixel table with loading percentage
def display_limited_rows_with_loading(pixel_table, num_rows=50):
    st.subheader(f"Pixel Values of the Processed Image (Showing {num_rows} rows)")
    st.write(pixel_table.head(num_rows))

# Function to display class probabilities in a table
def display_class_probabilities(predictions):
    prob_df = pd.DataFrame({'Cancer Class Category': class_labels, 'Probability (%)': predictions[0] * 100})
    st.table(prob_df.style.format({'Probability (%)': '{:.2f}%'}))

# Function to display spinner in the center
def display_spinner():
    spinner = st.spinner()
    spinner.text("Loading...")
    return spinner

# Streamlit app
st.title("Lung Cancer Detection")

# Sidebar navigation
page = st.sidebar.selectbox("Navbar", ["Prediction", "Performance Analysis", "Processed Pixels"])

if page == "Prediction":
    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose a test image...", type=["jpg", "jpeg", "png"])

    # Model performance analysis
    if uploaded_file is not None:
        # Display the uploaded image for training
        st.image(uploaded_file, caption="Uploaded Image (Training)", use_column_width=True)

        # Load and preprocess the test image
        st.write("Processing the image...")
        test_image = image.load_img(uploaded_file, target_size=(150, 150))
        st.image(test_image, caption="Processed Image (Training)", use_column_width=True)
        
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalize
        
        # Perform inference for prediction
        st.write("Performing inference...")

        # Add processing stage: Displaying intermediate layer activations
        intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[0].output)
        intermediate_output = intermediate_layer_model.predict(test_image)

        st.subheader("Intermediate Layer Activations")

        # Create an image with the desired colormap using Matplotlib
        fig, ax = plt.subplots()
        ax.imshow(intermediate_output[0, :, :, 0], cmap='viridis')
        ax.axis('off')
        st.pyplot(fig)

        predictions = model.predict(test_image)

        # Display probability scores for each class
        st.write("Class Probabilities:")
        display_class_probabilities(predictions)

        # Print the classification label with probability
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]
        predicted_class_probability = predictions[0][predicted_class_index] * 100
        st.success(f'Predicted Class: {predicted_class_label} with {predicted_class_probability:.2f}% probability')

elif page == "Performance Analysis":
    # Perform inference for performance analysis
    st.subheader("Model Performance Analysis")
    st.text("CNN Model Classification Report")
    st.image('cnn_classification_report.PNG', caption="CNN Model", use_column_width=True)
    st.text("SVM Model Classification Report")
    st.image('svm.PNG', caption="SVM Model", use_column_width=True)

    st.text("Model Accuracy")
    st.image('accuracy.PNG', caption="Model Accuracy", use_column_width=True)
    
    st.text("Model Loss")
    st.image('loss.PNG', caption="Model Loss", use_column_width=True)
    
    st.text("Performance Analysis For Normal Lungs")
    st.image('normal.png', caption="Normal Lungs", use_column_width=True)
    st.text("Performance Analysis For Large cell carcinoma Cancer")
    st.image('large.cell.carcinoma.png', caption="Large cell carcinoma Cancer", use_column_width=True)
    st.text("Performance Analysis For Squamous cell carcinoma Cancer")
    st.image('squamous.cell.carcinoma.png', caption="Squamous cell carcinoma Cancer", use_column_width=True)
    st.text("Performance Analysis For Adenocarcinoma Cancer")
    st.image('adenocarcinoma.png', caption="Adenocarcinoma Cancer", use_column_width=True)
    
    st.subheader("Model Confusion Matrix")
    st.image('confusion_matrix.png', caption="Confusion Matrix", use_column_width=True)

elif page == "Processed Pixels":
    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose a test image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image for training
        st.image(uploaded_file, caption="Uploaded Image (Training)", use_column_width=True)

        # Load and preprocess the test image
        st.write("Processing the image...")
        spinner = display_spinner()  # Display spinner for 5 seconds
        time.sleep(5)  # Add additional time if needed
        spinner.empty()  # Remove the spinner
        test_image = image.load_img(uploaded_file, target_size=(150, 150))
        st.image(test_image, caption="Processed Image (Training)", use_column_width=True)
        
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalize

        # Display a table showing pixel values
        pixel_table = pd.DataFrame(test_image.reshape(-1, 3), columns=['Red', 'Green', 'Blue'])
        display_limited_rows_with_loading(pixel_table)

        # Download button for CSV file
        st.download_button(
            label="Download Pixel Table as CSV",
            data=pixel_table.to_csv(index=False, encoding='utf-8'),
            file_name="pixel_table.csv",
            key="download_csv"
        )

