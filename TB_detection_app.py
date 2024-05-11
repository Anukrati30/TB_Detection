import streamlit as st
import torch
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import torchvision.models as models

def load_model(model_path, num_classes):
    model = models.densenet121(pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 1000)
    model.load_state_dict(torch.load(model_path))
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    model.eval()
    return model

def predict_image(image_path, model, classes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.sigmoid(model(img))
    predicted_class_index = torch.argmax(prediction, dim=1).item()
    predicted_class = classes[predicted_class_index]
    # Add condition for TB and Normal
    if predicted_class == 'TB':
        result = 'Positive'
        message = "You have tested positive for TB. It's important to consult a healthcare professional for further evaluation and treatment."
    else:
        result = 'Normal'
        message = "Your test results are normal. Continue to maintain good health practices."

    return result, message

def main():
    # HTML for centered title with emoji
    html_title = """
    <div style="text-align: center;">
        <h1>&#128080; TB Detection App</h1>
    </div>
    """
    st.markdown(html_title, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Upload Image")
    uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Load the model
        model_path = 'tb.pth' # Update with your model path
        classes = ['Normal', 'TB']
        model = load_model(model_path, len(classes))

        # Predict the image class and get message
        prediction, message = predict_image(uploaded_image, model, classes)

        # Display the prediction and message
        st.success(f"Prediction: {prediction}")
        st.info(message)

if __name__ == "__main__":
    main()
