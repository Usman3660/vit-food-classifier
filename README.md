# 🍔 Food-101 Image Classifier (ViT)

This repository contains the code for a Streamlit web application that classifies food images into one of 101 categories. It uses a **Vision Transformer (ViT)** model fine-tuned on the Food-101 dataset.

---

## 🚀 Deployed Demo

**[>> View the live demo here <<](https://vit-food-classifier.streamlit.app/#food-101-image-classifier)**

---

## 📋 Project Overview

This project's goal was to fine-tune a pre-trained Vision Transformer (`google/vit-base-patch16-224`) for a new downstream task: food classification. The model was trained in a Kaggle notebook and deployed as a public web app using Streamlit.

### Features
* **Image Upload**: Users can upload any `jpg`, `jpeg`, or `png` food image.
* **Live Classification**: The model provides a real-time prediction of the food category.
* **101 Categories**: The model can distinguish between 101 different food items, from Apple Pie to Waffles.

---

## 🤖 Model & Training

* **Base Model**: `google/vit-base-patch16-224`
* **Dataset**: [Food-101 Dataset](https://www.kaggle.com/datasets/kmader/food41)
* **Fine-Tuning**: The model was trained for 3 epochs on a 10% subset of the dataset (10,000 training images), achieving **85.2% accuracy** on the validation set.
* **Training Notebook**: [View the full Kaggle Training Notebook here](https) *(<-- Add link to your Kaggle Notebook)*
* **Hugging Face Model**: [View the fine-tuned model on the Hugging Face Hub](https) *(<-- Add link to your HF Model)*

---

## 💻 How to Run Locally

To run this application on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Usman3660/vit-food-classifier.git]
    cd vit-food-classifier
    ```

2.  **Install the requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
