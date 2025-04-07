# 🧴 AI-Powered Skincare Recommendation System

A deep learning-based skincare assistant that analyzes user skin images to detect acne and classify skin type, then recommends personalized skincare products using an integrated recommendation engine.

## 🚀 Overview

This AI system aims to assist individuals in identifying skin concerns and selecting the right skincare products based on their **acne condition** and **skin type**. It combines:

- 🔍 Acne Detection using **YOLOv5**
- 🧬 Skin Type Classification with **ResNet50**
- 💡 Product Recommendation System
- 🌐 Intuitive Gradio-based UI for real-time interaction

## 🎯 Key Features

- Detects acne types (blackheads, whiteheads, nodules, papules)
- Classifies skin type: **Oily, Dry, Normal, Combination**
- Recommends skincare products based on user-specific conditions
- Simple UI built with **Gradio** for quick, on-demand results

---

## 🎯 Use Case Diagram

Below is a conceptual use case diagram you can visualize:
       +-------------------+
       |   User Uploads    |
       |   Skin Image      |
       +--------+----------+
                |
                v
    +-----------+-----------+
    | ResNet50 (Skin Type   |
    | Classification)       |
    +-----------+-----------+
                |
                v
   +------------+------------+
   | YOLOv9 Model (Acne      |
   | Detection)              |
   +------------+------------+
                |
                v
 +--------------+---------------+
 | Product Recommender System   |
 +--------------+---------------+
                |
                v
       +--------+--------+
       | Gradio Interface |
       +------------------+

## 📦 Installation & Setup

Follow these steps to set up and run the system on your local machine:

```bash
# Clone the repository
git clone https://github.com/pavanmanikanta98/skincare-recommendation-system.git
cd skincare-recommendation-system

# Create Environment
-----------
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

```

## 📊 Results
Module	Accuracy / Score
Skin Type Classification	98% Train / 90% Test Accuracy
Acne Detection (YOLOv5)	65% mAP@0.5, 63% Recall

🔮 Future Plans

☁️ Deploy on cloud with API support for mobile apps.

📲 Integrate real-time camera feed support.

🧬 Incorporate additional skin conditions (e.g., rosacea, eczema).

📈 Add product review scoring and ratings from e-commerce APIs.

