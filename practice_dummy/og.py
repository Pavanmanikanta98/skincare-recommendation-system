import gradio as gr
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import joblib
from ultralytics import YOLO
import cv2 

# Generate recommendations considering both skin type & acne 
#TODO: should change this first
def generate_recommendations(df, model, skin_type, category, price_range=(0, 100), top_n=5):
    min_price, max_price = price_range
    category_df = df[(df['Label'] == category) & (df['Price'] >= min_price) & (df['Price'] <= max_price)]
    if category_df.empty:
        print(f"No products found for {category} in the price range {price_range}")
        return pd.DataFrame()
    
    probabilities = model.predict_proba(category_df[['Ingredients', 'Price']])
    skin_type_index = {'Oily': 0, 'Normal': 1, 'Dry': 2}[skin_type]
    suitability_scores = probabilities[0][:, 1]  # Probability of suitability

    category_df = category_df.assign(predicted_suitability=suitability_scores)

    
    recommendations = category_df.sort_values(by=['predicted_suitability', 'Rank'], ascending=False).head(top_n)
    
    return recommendations[['Brand', 'Name', 'Price', 'Rank', 'predicted_suitability']]


# Load Recommendation Model & Dataset
cosmetics_df = pd.read_csv("cosmetics.csv")
recommendation_model = joblib.load("recommendation_model.pkl")

# Define Skin Type Labels
index_label = {0: "Dry", 1: "Normal", 2: "Oily"}

# Define Acne Classes (as per YOLO model training)
acne_classes = ["blackhead", "nodule", "papule", "pustule", "whitehead"]

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_skin_model(path):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1000)  # 3 classes: Dry, Normal, Oily
    
    state_dict = torch.load(path, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}  # Remove unexpected keys

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("Skin Type Model Loaded Successfully.")
    return model

skin_model = load_skin_model("/home/pavan/Ds/pro/skincare product recommandation system/saved_skintype_model/best_model.pth")

# Load Acne Detection Model (YOLO)
acne_model = YOLO("/home/pavan/Ds/pro/skincare product recommandation system/acneDetection_model/best.pt")

# Define Transformations for Image Processing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to Predict Skin Type
def predict_skin_type(image):
    img = Image.fromarray(image).convert("RGB")
    img = transform(np.array(img))
    img = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = skin_model(img)
        pred = output.argmax(1).item()

   
    return index_label[pred]




def detect_acne(image, conf_threshold=0.3):
    # Ensure the image is a numpy array in RGB format.
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Run the YOLO model on the image.
    results = acne_model(image)
    
    detected_acne = set()  # Use a set to avoid duplicate detections

    # Check if the results list is empty.
    if len(results) == 0:
        print("No results returned from YOLO model.")
        return ["No Acne Detected"]
    
    # Loop through each result.
    for result in results:
        if not hasattr(result, "boxes"):
            print("Result has no boxes attribute.")
            continue

        print(f"Result has {len(result.boxes)} boxes.")
        for box in result.boxes:
            class_id = int(box.cls.item())  # Get class ID
            confidence = box.conf.item()      # Get confidence score
            print(f"Detected class {class_id} with confidence {confidence}")
            
            if confidence >= conf_threshold and class_id < len(acne_classes):
                detected_acne.add(acne_classes[class_id])
    
    detected_list = list(detected_acne) if detected_acne else ["No Acne Detected"]
    print("Detected acne types:", detected_list)
    return detected_list


# Function to Recommend Products Based on Skin Type & Acne Type
def recommend_products(image, category, min_price, max_price, top_n):
    skin_type = predict_skin_type(image)  # Predict skin type
    acne_types = detect_acne(image)  # Detect acne type(s)
    
    # Prepare filters
    price_range = (min_price, max_price)
    
    # Generate recommendations
    recommendations = generate_recommendations(
        cosmetics_df, recommendation_model, skin_type, category, price_range, top_n
    )
    
    if recommendations.empty:
        return pd.DataFrame({"Message": [f"No products found for {category} in the price range ${min_price} to ${max_price}."]}), skin_type, ", ".join(acne_types)
    
    return recommendations, skin_type, ", ".join(acne_types)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Skin Type & Acne Detection + Product Recommendation")
    gr.Markdown("Upload an image of your skin to detect skin type, acne type, and get personalized product recommendations!")

    image_input = gr.Image(type="numpy", label="Upload Skin Image")
    category_input = gr.Dropdown(label="Select Product Category", choices=['Moisturizer', 'Cleanser', 'Treatment', 'Face Mask', 'Eye Cream', 'Sun Protect'])
    min_price_input = gr.Number(label="Minimum Price", value=20, interactive=True)
    max_price_input = gr.Number(label="Maximum Price", value=50, interactive=True)
    top_n_input = gr.Slider(label="Number of Recommendations", minimum=1, maximum=10, value=5, step=1)

    recommendations_output = gr.Dataframe(label="Recommended Products")
    skin_type_output = gr.Textbox(label="Predicted Skin Type")
    acne_output = gr.Textbox(label="Detected Acne Type(s)")

    recommend_button = gr.Button("Get Recommendations")
    recommend_button.click(
        fn=recommend_products,
        inputs=[image_input, category_input, min_price_input, max_price_input, top_n_input],
        outputs=[recommendations_output, skin_type_output, acne_output]
    )

demo.launch()
