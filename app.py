# Import required libraries
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Custom CSS to change the background and other styles
st.markdown(
    """
    <style>
    /* Change the background color */
    .stApp {
        background-color: #F0F2F6;  /* Light gray background */
    }

    /* Style the title */
    .title {
        text-align: center;
        color: #FF4B4B;
        font-family: "Helvetica Neue", sans-serif;
        font-size: 2.5em;
        margin-bottom: 20px;
    }

    /* Style headers */
    h3 {
        color: #1F77B4;
        font-family: "Helvetica Neue", sans-serif;
    }

    /* Style buttons */
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 1em;
        font-family: "Helvetica Neue", sans-serif;
    }

    /* Style the footer */
    .footer {
        text-align: center;
        color: #7F7F7F;
        font-family: "Helvetica Neue", sans-serif;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set the title of the app with custom styling
st.markdown('<h1 class="title">Visual Question Answering</h1>', unsafe_allow_html=True)

# Function to load the BLIP model and processor
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return processor, model

# Load the BLIP model and processor
processor, model = load_model()

# Let the user choose between uploading an image or capturing from the camera
st.markdown('<h3>Choose an option to provide an image:</h3>', unsafe_allow_html=True)
option = st.radio("Select an option:", ("Upload an image", "Capture from camera"), key="option
