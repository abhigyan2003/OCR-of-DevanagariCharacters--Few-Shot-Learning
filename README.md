# 🕉️ Optical Character Recognition (OCR) of Devanagari Conjunct Characters using Few-Shot Learning

This project implements a **Few-Shot Learning (FSL)** approach for recognizing **handwritten Devanagari conjunct (compound) characters** using **prototype-based classification** with **VGG16 embeddings**.  
It aims to achieve efficient recognition with minimal training samples per class — a critical challenge for low-resource scripts like Devanagari.

---

## 🧠 Project Overview

Devanagari is a complex script with hundreds of conjunct (combined) characters that have limited labeled data available.  
To tackle this, we apply **Few-Shot Learning** by:
- Generating **support** and **query** sets for each class.
- Using a **pre-trained VGG16** model as an **embedding extractor**.
- Computing **class prototypes** from the support set.
- Classifying query images based on **Euclidean distance** from prototypes.

This approach minimizes training requirements while maintaining robust performance.

---

## 🧩 Methodology

### 1. Data Preparation
- Dataset is divided into `support` (2/3) and `query` (1/3) sets per class.
- Images are resized to `224x224` and stored for later embedding extraction.
- Valid image formats: `.jpg`, `.jpeg`, `.png`.

### 2. Embedding Extraction
- A **VGG16 (pretrained on ImageNet)** model is used without the top classification layers.
- The feature vector from the convolutional backbone serves as the **embedding** for each image.

### 3. Prototype Computation
- The **mean embedding** of all support samples of a class forms that class's prototype.
- This represents the centroid of each class in the embedding space.

### 4. Few-Shot Evaluation
- For each query image, the Euclidean distance to all class prototypes is computed.
- The query is assigned to the **nearest prototype class**.
- Accuracy, precision, recall, F1-score, and a confusion matrix are generated.





## 🧩 Acknowledgements

- **VGG16 pretrained model** — TensorFlow Keras Applications
- **Dataset**: Custom dataset of 1,890 handwritten Devanagari conjunct characters
- **Tools**: Python, TensorFlow, scikit-learn, Matplotlib, Seaborn

---


