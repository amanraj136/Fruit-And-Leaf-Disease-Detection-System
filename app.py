import streamlit as st
import tensorflow as tf
import numpy as np
import time
from PIL import Image

# ----------------------------- Load Model Once ---------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('f&v_leaf_detection_model.keras')

model = load_model()

# ------------------------- Class Names -----------------------------------------
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange_Citrus_Canker_Diseases_Leaf', 'Orange_Citrus_Nutrient_Deficiency_Yellow_Leaf_Orange',
    'Orange_Healthy_Leaf', 'Orange___Haunglongbing_(Citrus_greening)', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Sugarcane__Healthy', 'Sugarcane__Mosaic',
    'Sugarcane__RedRot', 'Sugarcane__Rust', 'Sugarcane__Yellow', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___healthy'
]

# ----------------------------- Prediction Function -----------------------------
def predict_disease(image_file):
    image = Image.open(image_file).convert('RGB')
    image = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# ----------------------------- Pesticide Suggestion ----------------------------
def get_pesticide_suggestion(class_name):
    pesticides = {
        "Apple___Apple_scab": ["Captan", "Mancozeb", "Thiophanate-methyl"],
        "Apple___Black_rot": ["Bordeaux mixture", "Captan", "Ziram"],
        "Apple___Cedar_apple_rust": ["Captan", "Mancozeb", "Thiophanate-methyl"],
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ["Chlorothalonil", "Azoxystrobin", "Propiconazole"],
        "Corn_(maize)___Common_rust_": ["Chlorothalonil", "Triazole fungicides"],
        "Corn_(maize)___Northern_Leaf_Blight": ["Azoxystrobin", "Pyraclostrobin", "Propiconazole"],
        "Grape___Black_rot": ["Bordeaux mixture", "Mancozeb", "Captan"],
        "Grape___Esca_(Black_Measles)": ["Propiconazole", "Trifloxystrobin", "Boscalid"],
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ["Mancozeb", "Captan", "Thiophanate-methyl"],
        "Orange_Citrus_Canker_Diseases_Leaf": ["Copper-based fungicides", "Streptomycin sulfate"],
        "Orange_Citrus_Nutrient_Deficiency_Yellow_Leaf_Orange": ["Nitrogen", "Potassium", "Magnesium fertilizers"],
        "Potato___Early_blight": ["Chlorothalonil", "Mancozeb", "Copper-based fungicides"],
        "Potato___Late_blight": ["Chlorothalonil", "Mancozeb", "Metalaxyl"],
        "Tomato___Bacterial_spot": ["Copper-based fungicides", "Streptomycin sulfate"],
        "Tomato___Early_blight": ["Chlorothalonil", "Mancozeb", "Copper-based fungicides"],
        "Tomato___Late_blight": ["Chlorothalonil", "Mancozeb", "Metalaxyl"],
        "Sugarcane__RedRot": ["Chlorothalonil", "Mancozeb", "Copper-based fungicides"]
    }
    return "\n".join([f"{i+1}. {item}" for i, item in enumerate(pesticides.get(class_name, ["No pesticide needed. Leaf is Healthy!"]))])

# ----------------------------- Sidebar Navigation -----------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Leaf Disease Recognition"])

# ----------------------------- Home Page --------------------------------------
if app_mode == "Home":
    st.header("Fruit & Vegetable Leaf Disease Recognition System")
    st.image("asset/home_plant.jpg", width= 500)
    st.markdown("""
    Welcome to the Fruits & Vegetables Leaf Disease Recognition System! 🌿🔍
    
    Our aim is to help in identifying leaf diseases efficiently. Upload an image of a fruit or vegetable leaf, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a leaf with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Leaf Disease Recognition** page in the sidebar to upload an image and experience the power of our Fruit & Vegetable Leaft Disease Recognition System!
    """)

# ----------------------------- About Page -------------------------------------
elif app_mode == "About":
    st.header("About")
    st.markdown("""
### Dataset Information

The dataset used for this project contains images of fruits and vegetables leaves with various diseases and healthy leaves. It was augmented and reorganized into training, validation and testing sets.

- Training Set: 70,597 images
- Validation Set: 42,532 images
- Test Set: 33 images

The dataset contains 28 different classes of plant leaf diseases and healthy leaves. The images are labeled and categorized for training a deep learning model.
    """)

# -------------------------- Disease Recognition Page --------------------------
elif app_mode == "Leaf Disease Recognition":
    st.header("Leaf Disease Recognition")
    test_image = st.file_uploader("📤 Upload a Leaf Image", type=["jpg", "jpeg", "png"])

    if test_image:
        # st.image(test_image, use_column_width=True, caption="Uploaded Image")
        st.image(test_image, width=150, caption="Uploaded Image")

    if st.button("🔍 Predict"):
        if not test_image:
            st.error("⚠ Please upload an image before prediction.")
        else:
            with st.spinner("⏳ Analyzing the image..."):
                time.sleep(1)
                result_index = predict_disease(test_image)
                class_name = CLASS_NAMES[result_index]

                st.success(f"🦠 **Detected Disease:** `{class_name}`")
                st.write("💊 **Pesticide Suggestions:**")
                st.info(get_pesticide_suggestion(class_name))
