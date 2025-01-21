import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import time


def model_prediction(test_image, model):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

def load_model_from_upload():
    model_file = st.file_uploader("Upload your model", type="h5")
    if model_file is not None:
        model = tf.keras.models.load_model(BytesIO(model_file.read()))
        st.success("Model loaded successfully!")
        return model
    return None

# In your main Streamlit app logic
model = load_model_from_upload()


def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

def submit_feedback(name, email, feedback):
    url = "https://api.web3forms.com/submit"
    payload = {
        "access_key": "4ffcbd0a-8334-41a7-af0a-d8552c02dd27",
        "name": name,
        "email": email,
        "message": feedback
    }
    response = requests.post(url, data=payload)
    return response

def feedback():
    st.title("Feedback / Suggestions")
    st.markdown("We value your feedback! Please provide your suggestions or feedback below:")

    with st.form(key='feedback_form', clear_on_submit=True):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        feedback_text = st.text_area("Your Feedback/Suggestions", height=150)
        
        submit_feedback_btn = st.form_submit_button(label='Submit Feedback')
        
        if submit_feedback_btn:
            if name and email and feedback_text:
                response = submit_feedback(name, email, feedback_text)
                if response.status_code == 200:
                    st.success("Your feedback has been submitted successfully! Thank you.")
                else:
                    st.error("Failed to submit feedback. Please try again later.")
            else:
                st.warning("Please fill out all fields.")

def about():
    st.title("About Crop Disease Identification")
    st.markdown("""
    ### Introduction
    The **Crop Disease Identification** system leverages advanced deep learning techniques to identify diseases in various crops from images. This helps farmers and agricultural experts in early detection and management of crop diseases, ensuring better yield and quality.

    ### How It Works
    1. **Image Upload**: Users can upload an image of the crop they want to analyze.
    2. **Disease Detection**: Our convolutional neural network (CNN) model processes the image to detect and identify the crop disease.
    3. **Result Display**: The identified disease is displayed along with the corresponding crop name.

    ### Benefits
    - **Early Detection**: Helps in early detection of crop diseases, allowing for timely intervention.
    - **Improved Yield**: By managing diseases effectively, crop yield and quality can be significantly improved.
    - **User-Friendly**: Easy-to-use interface for farmers and agricultural experts.

    ### Note
    Our model is trained on specific crops like apple, blueberry, cherry, corn, grape, orange, peach, pepper, potato, raspberry, soybean, squash, strawberry, and tomato. Uploading images of other crops, such as spinach, may result in incorrect predictions. Please use images of the listed crops for accurate results.
    """, unsafe_allow_html=True)

# Set TensorFlow logging level to 'ERROR' to suppress warnings and info messages
tf.get_logger().setLevel('ERROR')

st.set_page_config(
    page_title="Advanced Crop Disease Solutions",
    page_icon="🌾",
    layout="wide",
)

# Sidebar
st.sidebar.markdown("""
    <style>
        .sidebar-title {
            font-size: 24px;
            font-weight: bold;
            background: linear-gradient(270deg, #8FBC8F, #2E8B57);
            -webkit-background-clip: text;
            color: transparent;
            text-align: center;
        }
    </style>
    <div class="sidebar-title">KrishiGyaan</div>
""", unsafe_allow_html=True)

app_mode = st.sidebar.radio("Select View", ["HOME", "DISEASE RECOGNITION", "ABOUT", "FEEDBACK"])
# Main Page
if app_mode == "HOME":
    st.markdown("""
        <style>
            .header {
                font-size: 36px;
                text-align: center;
                background: linear-gradient(270deg, #3C8D40, #6A9B4D);
                font-family: 'Arial', sans-serif;
                font-weight: bold;
                margin-top: 30px;
                margin-bottom: 20px;
            }
            .container {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .title {
                font-size: 24px;
                font-weight: 600;
            }
            .image-section {
                margin-top: 20px;
            }
        </style>
        <div class="container">
            <div class="title">KrishiGyaan</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='header'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)
    # Import Image from Pillow to open images
    from PIL import Image
    img = Image.open("crop_dis_identify.jpg")
    img = img.resize((900, 503))

    # Display image using streamlit
    st.image(img, caption="Smart Disease Detection", use_container_width=True, width=100)

elif app_mode == "DISEASE RECOGNITION":
    st.markdown("<h2 class='header'>DISEASE RECOGNITION</h2>", unsafe_allow_html=True)
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
    
    if test_image:
        st.image(test_image, width=250, use_container_width=True)
    
    # Predict button
    if st.button("Predict"):
        with st.spinner('Processing...'):
            time.sleep(2)
        st.success('Done!')

        st.write("Our Prediction:")
        result_index = model_prediction(test_image)
        
        # Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        
        st.success(f"Model is predicting it's a: {class_name[result_index]}")

elif app_mode == "ABOUT":
    about()

elif app_mode == "FEEDBACK":
    feedback()
