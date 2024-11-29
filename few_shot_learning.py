import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Function to create directories
def create_directories(output_dir, classes):
    support_dir = os.path.join(output_dir, 'support')
    query_dir = os.path.join(output_dir, 'query')
    os.makedirs(support_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)
    
    for cls in classes:
        os.makedirs(os.path.join(support_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(query_dir, cls), exist_ok=True)

# Function to load and save images in support and query folders
def load_and_save_images(directory, target_size, output_dir, support_ratio=2/3):
    support_images, query_images, support_labels, query_labels = [], [], [], []
    class_labels = sorted(os.listdir(directory))
    label_map = {label: class_name for label, class_name in enumerate(class_labels)}
    print("Label Map:", label_map)
    
    create_directories(output_dir, class_labels)
    
    total_support_images = 0
    total_query_images = 0

    valid_extensions = ('.jpeg', '.jpg', '.png')
    
    for label, class_name in enumerate(class_labels):
        class_path = os.path.join(directory, class_name)
        images = [img for img in os.listdir(class_path) if img.lower().endswith(valid_extensions)]
        np.random.shuffle(images)
        
        num_support = int(len(images) * support_ratio)
        support_imgs = images[:num_support]
        query_imgs = images[num_support:]
        
        total_support_images += len(support_imgs)
        total_query_images += len(query_imgs)
        
        for img in support_imgs:
            img_path = os.path.join(class_path, img)
            try:
                img_data = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img_data)
                support_images.append(img_array)
                support_labels.append(label)
                img_data.save(os.path.join(output_dir, 'support', class_name, img))
            except Exception as e:
                print(f"Skipping file {img_path}: {e}")
        
        for img in query_imgs:
            img_path = os.path.join(class_path, img)
            try:
                img_data = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img_data)
                query_images.append(img_array)
                query_labels.append(label)
                img_data.save(os.path.join(output_dir, 'query', class_name, img))
            except Exception as e:
                print(f"Skipping file {img_path}: {e}")
    
    print(f"Total support images loaded: {total_support_images}")
    print(f"Total query images loaded: {total_query_images}")
    
    return (np.array(support_images), np.array(support_labels), np.array(query_images), np.array(query_labels), class_labels)

# Function to build the embedding model
def build_embedding_model(input_shape):
    base_model = VGG16(include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model

    input_layer = Input(shape=input_shape)
    x = base_model(input_layer)
    x = Flatten()(x)
    embedding_model = Model(input_layer, x, name="embedding_model")
    return embedding_model

# Function to calculate prototypes for each class
def compute_prototypes(embedding_model, support_images, support_labels, num_classes, batch_size=32):
    embeddings = embedding_model.predict(support_images, batch_size=batch_size)
    prototypes = []
    for class_id in range(num_classes):
        class_embeddings = embeddings[support_labels == class_id]

        # Check if there are embeddings for the current class
        if len(class_embeddings) > 0:
            prototype = np.mean(class_embeddings, axis=0)
        else:
            prototype = np.zeros(embeddings.shape[1])  # Create a zero-vector for empty classes

        prototypes.append(prototype)

    return np.array(prototypes)

# Function to calculate pairwise distances
def pairwise_distances(a, b):
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    a = tf.expand_dims(a, axis=1)
    b = tf.expand_dims(b, axis=0)
    return tf.reduce_sum(tf.square(a - b), axis=-1)

# Function to evaluate the model on query set using prototypes
def evaluate_few_shot(embedding_model, query_images, query_labels, prototypes, class_labels, batch_size=32):
    query_embeddings = embedding_model.predict(query_images, batch_size=batch_size)
    distances = pairwise_distances(query_embeddings, prototypes)
    predictions = np.argmin(distances, axis=1)
    
    accuracy = accuracy_score(query_labels, predictions)
    precision = precision_score(query_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(query_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(query_labels, predictions, average='macro', zero_division=0)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(query_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Classification Report
    report = classification_report(query_labels, predictions, target_names=class_labels, zero_division=0)
    print('Classification Report:\n', report)
    
    return accuracy

# Define your directories and parameters
root_dir = "/kaggle/input/devanagari-consonant-conjuncts-fsl/main"
target_size = (224, 224)
output_dir = "/kaggle/working/"
num_classes = len(os.listdir(root_dir))

# Load support and query sets and save images
train_support_images, train_support_labels, train_query_images, train_query_labels, class_labels = load_and_save_images(
    root_dir, target_size, output_dir, support_ratio=2/3
)

# Build the embedding model
embedding_model = build_embedding_model(input_shape=(224, 224, 3))

# Compute prototypes for each class
prototypes = compute_prototypes(embedding_model, train_support_images, train_support_labels, num_classes)

# Evaluate the model on query set
evaluate_few_shot(embedding_model, train_query_images, train_query_labels, prototypes, class_labels)
