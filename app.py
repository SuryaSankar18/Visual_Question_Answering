# Import required libraries
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Set the title of the app
st.title("Visual Question Answering with BLIP Model")

# Function to load the BLIP model and processor
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return processor, model

# Load the BLIP model and processor
processor, model = load_model()

# Let the user choose between uploading an image or capturing from the camera
st.write("### Choose an option to provide an image:")
option = st.radio("Select an option:", ("Upload an image", "Capture from camera"))

# Initialize the image variable
image = None

# Handle image upload
if option == "Upload an image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

# Handle camera capture
elif option == "Capture from camera":
    st.write("### Capture an image from your camera:")
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image)

# Display the image if available
if image is not None:
    st.write("### Image:")
    st.image(image, caption="Uploaded/Captured Image", use_column_width=True)

    # Ask a question about the image
    question = st.text_input("Ask a question about the image:", "What is the color of the shirt?")

    # Process the image and generate an answer
    if st.button("Get Answer"):
        if question.strip() == "":
            st.error("Please enter a question.")
        else:
            # Preprocess the image and question
            inputs = processor(image, question, return_tensors="pt")

            # Generate an answer using the BLIP model
            st.write("### Answer:")
            with st.spinner("Processing image and generating an answer..."):
                out = model.generate(**inputs)
                answer = processor.decode(out[0], skip_special_tokens=True)
                st.success(answer)
else:
    st.warning("Please upload an image or capture one from the camera.")