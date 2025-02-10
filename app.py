# Import required libraries
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Custom CSS to set an image as the background
st.markdown(
    """
    <style>
    /* Set an image as the background */
    .stApp {
        background-image: url("https://img.freepik.com/free-photo/modern-background-with-lines_1361-3533.jpg");
        background-size: cover;  /* Cover the entire app */
        background-position: center;  /* Center the image */
        background-repeat: no-repeat;  /* Prevent repeating */
        background-attachment: fixed;  /* Fix the background while scrolling */
    }

    /* Add a semi-transparent overlay to improve readability */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(255, 255, 255, 0.7);  /* White overlay with 70% opacity */
        z-index: -1;  /* Place the overlay behind the content */
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
option = st.radio("Select an option:", ("Upload an image", "Capture from camera"), key="option")

# Initialize the image variable
image = None

# Handle image upload
if option == "Upload an image":
    st.markdown('<h3 style="color: #2CA02C;">Upload an image</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="upload")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

# Handle camera capture
elif option == "Capture from camera":
    st.markdown('<h3 style="color: #D62728;">Capture an image from your camera</h3>', unsafe_allow_html=True)
    camera_image = st.camera_input("Take a picture", key="camera")
    if camera_image is not None:
        image = Image.open(camera_image)

# Display the image if available
if image is not None:
    st.markdown('<h3 style="color: #9467BD;">Image:</h3>', unsafe_allow_html=True)
    st.image(image, caption="Uploaded/Captured Image", use_column_width=True)

    # Ask a question about the image
    st.markdown('<h3 style="color: #8C564B;">Ask a question about the image:</h3>', unsafe_allow_html=True)
    question = st.text_input("Enter your question here", key="question")

    # Process the image and generate an answer
    if st.button("Get Answer", key="answer_button"):
        if question.strip() == "":
            st.error("Please enter a question.")
        else:
            # Preprocess the image and question
            inputs = processor(image, question, return_tensors="pt")

            # Generate an answer using the BLIP model
            st.markdown('<h3 style="color: #E377C2;">Answer:</h3>', unsafe_allow_html=True)
            with st.spinner("Processing image and generating an answer..."):
                out = model.generate(**inputs)
                answer = processor.decode(out[0], skip_special_tokens=True)
                st.success(f"**Answer:** {answer}")
else:
    st.warning("Please upload an image or capture one from the camera.")

# Add a colorful footer
st.markdown(
    """
    <div class="footer">
        <hr>
        <p>Powered by Streamlit and BLIP 🤖</p>
    </div>
    """,
    unsafe_allow_html=True
)
