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

---

## 📁 Directory Structure

```
📦 devanagari-fsl
├── main/                    # Root dataset directory
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── output/                  # Generated during execution
│   ├── support/
│   ├── query/
├── few_shot_ocr.py          # Main script
└── README.md                # Project documentation
```

---

## ⚙️ Requirements

Make sure you have the following dependencies installed:

```bash
pip install tensorflow numpy scikit-learn matplotlib seaborn
```

Or install all at once using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1: Set up dataset paths
Modify the following in the script:

```python
root_dir = "/kaggle/input/devanagari-consonant-conjuncts-fsl/main"
output_dir = "/kaggle/working/"
target_size = (224, 224)
```

### Step 2: Run the main script
```bash
python few_shot_ocr.py
```

### Step 3: Outputs
- Generated support/query directories.
- Confusion Matrix and Classification Report.
- Printed evaluation metrics (Accuracy, Precision, Recall, F1-Score).

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correctly classified query samples |
| **Precision** | Class-wise average correctness of predictions |
| **Recall** | Ability to detect all samples of a class |
| **F1 Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Visual representation of predictions vs actual labels |

---

## 🧪 Sample Results

Example outputs during evaluation:

```
Accuracy: 94.67%
Precision: 93.45%
Recall: 94.12%
F1 Score: 93.78%
```

Confusion matrix and classification report are automatically displayed and saved.

---

## 🧠 Key Learnings

- Demonstrated how prototype-based Few-Shot Learning can effectively classify complex low-resource scripts.
- Leveraged VGG16 embeddings for transfer learning without full retraining.
- Established a scalable and interpretable baseline for Devanagari OCR research.

---

## 🧾 Citation / Reference

If you use this work in your research, please cite:

```
Borah, Abhigyan. "Optical Character Recognition (OCR) of Devanagari Conjunct Characters using Few-Shot Learning", 
B.Tech Minor Project, Sikkim Manipal Institute of Technology, 2025.
```

---

## 👨‍💻 Author

**Abhigyan Borah**  
B.Tech in Computer Science and Engineering (AI Minor)  
Sikkim Manipal Institute of Technology, India

- 🔗 [LinkedIn](https://www.linkedin.com/in/abhigyan-borah)
- 💻 [GitHub](https://github.com/abhigyanborah)
- 📧 abhigyanborah3@gmail.com

---

## 🧩 Acknowledgements

- **VGG16 pretrained model** — TensorFlow Keras Applications
- **Dataset**: Custom dataset of 1,890 handwritten Devanagari conjunct characters
- **Tools**: Python, TensorFlow, scikit-learn, Matplotlib, Seaborn

---

## ⭐ Future Work

- Extend to Transformer-based meta-learning approaches (e.g., Meta-BERT, ViT-FSL).
- Incorporate contrastive or triplet loss for improved embedding separation.
- Benchmark against Prototypical Networks and Matching Networks.

---

## 📄 License

```
MIT License © 2025 Abhigyan Borah
```
