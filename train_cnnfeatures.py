
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the directory where your script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define paths to the directories where your training and test data are located
train_data_dir = os.path.join(script_directory, 'train_data')
test_data_dir = os.path.join(script_directory, 'test_data')
hog_train_dir = os.path.join(script_directory, 'HOG_train')
hog_test_dir = os.path.join(script_directory, 'HOG_test')

print("Loading and preprocessing training images and annotations...")

# Load and preprocess training images and annotations
train_images = []
train_annotations = []

for class_folder in os.listdir(train_data_dir):
    class_folder_path = os.path.join(train_data_dir, class_folder)
    if os.path.isdir(class_folder_path):
        print(f"Processing class folder: {class_folder_path}")

        annotation_file = None
        for file_name in os.listdir(class_folder_path):
            if file_name.endswith('.csv'):
                annotation_file = os.path.join(class_folder_path, file_name)
                break

        if annotation_file is None:
            print(f"Annotation file not found for class {class_folder}")
            continue

        print(f"Loading annotations from: {annotation_file}")
        annotations_df = pd.read_csv(annotation_file, delimiter=';')  # Load annotations from CSV

        for index, row in annotations_df.iterrows():
            image_file = row['Filename']
            print(f"Processing image: {image_file}")
            image_path = os.path.join(class_folder_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load PPM image using OpenCV
            train_images.append(image)

            annotation = row['ClassId']
            train_annotations.append(annotation)

print("Total training images:", len(train_images))
print("Total training annotations:", len(train_annotations))

print("Preprocessing training images and annotations...")

# Preprocess training images and annotations
image_size = (32, 32)  # Your desired image size

train_images_preprocessed = [cv2.resize(img, image_size) for img in train_images]
train_images_preprocessed = np.array(train_images_preprocessed) / 255.0
train_annotations = np.array(train_annotations)

print("Loading and preprocessing HOG features for training...")

# Load and preprocess HOG features for training
train_hog_features = []
train_labels = []

for hog_folder in os.listdir(hog_train_dir):
    hog_folder_path = os.path.join(hog_train_dir, hog_folder)
    if os.path.isdir(hog_folder_path):
        class_label = int(hog_folder.split('_')[1])  # Extract the class label from folder name
        
        print(f"Processing HOG folder: {hog_folder_path}")
        print(f"Class label: {class_label}")

        for image_file in os.listdir(hog_folder_path):
            if image_file.endswith('.txt'):
                text_file_path = os.path.join(hog_folder_path, image_file)
                with open(text_file_path, 'r') as f:
                    hog_features = [float(val) for val in f.read().split()]
                train_hog_features.append(hog_features)
                train_labels.append(class_label)

train_hog_features = np.array(train_hog_features)
train_labels = np.array(train_labels)

print("Total training HOG features:", len(train_hog_features))
print("Total training labels:", len(train_labels))

# Build the CNN model (LeNet-5 architecture)
model = Sequential([
    Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(16, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(43, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
model.fit([train_images_preprocessed, train_hog_features], train_annotations, epochs=10, batch_size=32, validation_split=0.2)

model_save_path = os.path.join(script_directory, 'trained_model_with_hog.h5')
model.save(model_save_path)
print(f"Trained model with HOG features saved at: {model_save_path}")
