import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# --- Page Config ---
st.set_page_config(
    page_title="🍔 Food-101 Classifier",
    page_icon="🍔",
    layout="centered",
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the fine-tuned model and processor from Hugging Face."""

   
    model_id = "Usman366/vit-base-patch16-224-food101" 

    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {e}\n\nMake sure you've pushed your model to the Hub and set the correct `model_id`.")
        return None, None

processor, model = load_model()

# --- Helper Function ---
def predict(image, model, processor):
    """Generates a prediction for the uploaded image."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the top prediction
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    # Clean up label name (e.g., "hot_dog" -> "Hot Dog")
    predicted_class_clean = predicted_class.replace("_", " ").title()

    return predicted_class_clean

# --- Streamlit UI ---
st.title("🍔 Food-101 Image Classifier")
st.markdown("Upload an image of food, and the Vision Transformer (ViT) model will predict what it is!")

uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Please upload an image to get started.")
elif model is None:
    # This message is shown if the model loading failed
    pass 
else:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Prediction button
    if st.button('Classify Food!'):
        with st.spinner('Thinking...'):
            prediction = predict(image, model, processor)
            st.success(f"**Prediction: {prediction}** 🍕")

st.markdown("---")
st.markdown(
    "Powered by a `vit-base-patch16-224` model fine-tuned on the Food-101 dataset. "
    "\n[View on GitHub](Your_GitHub_Repo_Link)" # <-- Add your GitHub link
)
