# Import required libraries
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Set the title of the app with custom styling
st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B; font-family: "Helvetica Neue", sans-serif;'>
        Visual Question Answering
    </h1>
    """,
    unsafe_allow_html=True
)

# Function to load the BLIP model and processor
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return processor, model

# Load the BLIP model and processor
processor, model = load_model()

# Let the user choose between uploading an image or capturing from the camera
st.markdown(
    """
    <h3 style='color: #1F77B4; font-family: "Helvetica Neue", sans-serif;'>
        Choose an option to provide an image:
    </h3>
    """,
    unsafe_allow_html=True
)

option = st.radio("Select an option:", ("Upload an image", "Capture from camera"), key="option")

# Initialize the image variable
image = None

# Handle image upload
if option == "Upload an image":
    st.markdown(
        """
        <h4 style='color: #2CA02C; font-family: "Helvetica Neue", sans-serif;'>
            Upload an image
        </h4>
        """,
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="upload")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

# Handle camera capture
elif option == "Capture from camera":
    st.markdown(
        """
        <h4 style='color: #D62728; font-family: "Helvetica Neue", sans-serif;'>
            Capture an image from your camera
        </h4>
        """,
        unsafe_allow_html=True
    )
    camera_image = st.camera_input("Take a picture", key="camera")
    if camera_image is not None:
        image = Image.open(camera_image)

# Display the image if available
if image is not None:
    st.markdown(
        """
        <h3 style='color: #9467BD; font-family: "Helvetica Neue", sans-serif;'>
            Image:
        </h3>
        """,
        unsafe_allow_html=True
    )
    st.image(image, caption="Uploaded/Captured Image", use_column_width=True)

    # Ask a question about the image
    st.markdown(
        """
        <h3 style='color: #8C564B; font-family: "Helvetica Neue", sans-serif;'>
            Ask a question about the image:
        </h3>
        """,
        unsafe_allow_html=True
    )
    question = st.text_input("", key="question")

    # Process the image and generate an answer
    if st.button("Get Answer", key="answer_button"):
        if question.strip() == "":
            st.error("Please enter a question.")
        else:
            # Preprocess the image and question
            inputs = processor(image, question, return_tensors="pt")

            # Generate an answer using the BLIP model
            st.markdown(
                """
                <h3 style='color: #E377C2; font-family: "Helvetica Neue", sans-serif;'>
                    Answer:
                </h3>
                """,
                unsafe_allow_html=True
            )
            with st.spinner("Processing image and generating an answer..."):
                out = model.generate(**inputs)
                answer = processor.decode(out[0], skip_special_tokens=True)
                st.success(f"**Answer:** {answer}")
else:
    st.warning("Please upload an image or capture one from the camera.")

# Add a colorful footer
st.markdown(
    """
    <div style='text-align: center; color: #7F7F7F; font-family: "Helvetica Neue", sans-serif; margin-top: 50px;'>
        <hr>
        <p>Powered by Streamlit and BLIP 🤖</p>
    </div>
    """,
    unsafe_allow_html=True
)
