from tensorflow.keras.applications import (
    EfficientNetB0, InceptionResNetV2, MobileNetV2, VGG16
)
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_pre
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_pre
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
import os
from imutils import paths
import matplotlib.pyplot as plt

models_config = {
    "EfficientNetB0" : (EfficientNetB0,efficientnet_pre),
    "VGG16" : (VGG16,vgg_pre),
    "InceptionResNetV2" : (InceptionResNetV2,inception_pre),
    "MobileNetV2" : (MobileNetV2,mobilenet_pre)
}

def load_and_preprocess_data(image_paths, target_size, preprocess_func):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img,target_size)
        img = img.astype("float32")
        img = preprocess_func(img)
        images.append(img)
    return np.array(images)

def feature_extractor(model_class,preprocess_func,image_data,input_shape):
    base_model = model_class(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Model(inputs=base_model.input, outputs=base_model.output)
    # Extract features
    features = model.predict(image_data, batch_size=32, verbose=1)

    # Flatten the features
    features_flatten = features.reshape(features.shape[0], -1)

    print(features_flatten.shape)

    data = features_flatten
    return data,model

def find_top_k_similar_images(dataset_paths, query_embedding, dataset_embeddings, top_k=10):
    dataset_embeddings_norm = tf.nn.l2_normalize(dataset_embeddings, axis=1)
    query_embedding_norm = tf.nn.l2_normalize(tf.reshape(query_embedding, (1, -1)), axis=1)
    similarities = tf.linalg.matmul(query_embedding_norm, dataset_embeddings_norm, transpose_b=True).numpy()[0]
    top_k_idx = similarities.argsort()[-top_k:][::-1]
    return [(dataset_paths[i], similarities[i]) for i in top_k_idx]


def plot_top_k_images(query_img_path, image_paths, results, title="Top Similar Images", top_k = 10):
    """
    Plots the query image and top-k similar images.

    :param top_images: List of (image_path, similarity) tuples
    :param query_img_path: Path to the query image
    """
    num_models = len(results)

    plt.figure(figsize=(2.5 * (top_k + 1),3 * num_models))
    
    query_img = cv2.imread(query_img_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    
    for row_idx, (model_name,top_images) in enumerate(results.items()):
        plt.subplot(num_models, top_k + 1, row_idx * (top_k+1) + 1)
        plt.imshow(query_img)
        plt.title(f"{model_name}\nQuery")
        plt.axis("off")
        
        for i, (img_path, similarity) in enumerate(top_images):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(num_models, top_k + 1, row_idx * (top_k+1) + i + 2)
            plt.imshow(img)
            plt.title(f"Sim: {similarity:.2f}")
            plt.axis("off")

    plt.suptitle(title, fontsize=18)
    #plt.tight_layout()
    plt.show()

results = {}
target_size = (224,224)
input_shape = (224,224,3)
image_dir = "Dataset_BUSI_with_GT"
image_paths = list(paths.list_images(image_dir))
query_img_path = r"C:\Users\hrish\Desktop\project\Dataset_BUSI_with_GT\malignant\malignant (14).png"

for model_name, (model_class,preprocess_func) in models_config.items():
    print(f"[INFO] Processing with {model_name}")

    image_data = load_and_preprocess_data(image_paths,target_size,preprocess_func)
    features, model = feature_extractor(model_class,preprocess_func,image_data,input_shape)

    query_img = cv2.imread(query_img_path)
    query_img = cv2.resize(query_img,target_size).astype("float32")
    query_img = preprocess_func(query_img)
    query_img = np.expand_dims(query_img,axis=0)
    query_features = model.predict(query_img)
    query_features_flatten = query_features.reshape(1,-1)

    top_images = find_top_k_similar_images(image_paths, query_features_flatten, features)
    results[model_name] = top_images


plot_top_k_images(query_img_path,image_paths,results)