import os
import cv2
import time
import pickle
import numpy as np
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
# Clear session
tf.keras.backend.clear_session()
tf.config.run_functions_eagerly(True)
print("Eager execution enabled:", tf.executing_eagerly())
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import preprocess_input
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig


class DataPreprocessor:
    def __init__(self, training_config):
        """
        Handles data loading and preprocessing based on the training configuration.
        """
        self.training_config = training_config


    def load_data(self):
        """
        Load and preprocess the training and testing data.
        """
        X, Y = [], []

        # Load training data
        for label in self.training_config.labels:
            folder_path = os.path.join(self.training_config.training_data, label)
            for img_name in os.listdir(folder_path):
                img = cv2.imread(os.path.join(folder_path, img_name))
                img = cv2.resize(img, (224, 224))
                X.append(img)
                Y.append(self.training_config.labels.index(label))  # Map label to numeric value

        # Load testing data
        for label in self.training_config.labels:
            folder_path = os.path.join(self.training_config.testing_data, label)
            for img_name in os.listdir(folder_path):
                img = cv2.imread(os.path.join(folder_path, img_name))
                img = cv2.resize(img, (224, 224))
                X.append(img)
                Y.append(self.training_config.labels.index(label))  # Map label to numeric value

        # Convert to NumPy arrays and preprocess
        X = np.array(X)
        Y = np.array(Y)        

        X = preprocess_input(X)
        # Debug shapes
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")

        
        return X, Y


    def prepare_data(self, X, Y, test_size=0.2):
        """
        Split the data into training and validation sets and one-hot encode the labels.
        """
        # Split data
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=42)

        # One-hot encode the labels
        num_classes = len(self.training_config.labels)
        Y_train = to_categorical(Y_train, num_classes=num_classes)
        Y_val = to_categorical(Y_val, num_classes=num_classes)

        # Debug shapes
        print(f"X_train shape: {X_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"Y_train shape: {Y_train.shape}")
        print(f"Y_val shape: {Y_val.shape}")

        # Create the save directory if it doesn't exist
        os.makedirs(self.training_config.split_dir, exist_ok=True)

        # Save the datasets using pickle
        with open(os.path.join(self.training_config.split_dir, "train_data.pkl"), "wb") as f:
            pickle.dump((X_train, Y_train), f)
        with open(os.path.join(self.training_config.split_dir, "val_data.pkl"), "wb") as f:
            pickle.dump((X_val, Y_val), f)

        return X_train, Y_train, X_val, Y_val, num_classes

class Training:
    def __init__(self, training_config: TrainingConfig,learning_rate, X_train, Y_train, X_val, Y_val,):
        """
        Initialize the training process with configuration and preprocessed data.
        """
        self.config = training_config
        self.learning_rate = learning_rate
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.model = None  # Placeholder for the loaded model

    def load_base_model(self):
        print(f"Loading base model from: {self.config.updated_base_model_path}")
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        print("Base model loaded successfully.")

        # Recompile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        print("Model recompiled successfully.")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the trained model to the specified path.
        """
        model.save(path)
        print(f"Model saved successfully at: {path}")

    def train(self):
        """
        Train the loaded model using preprocessed datasets.
        """
        import tensorflow.keras.backend as K
        K.clear_session()

        if self.model is None:
            raise ValueError("Model not loaded. Call 'load_base_model()' before training.")

        # Train the model
        print("Starting training...")

        history = self.model.fit(
            self.X_train, 
            self.Y_train,
            validation_data=(self.X_val, self.Y_val),
            epochs=self.config.params_epochs,
            batch_size=self.config.params_batch_size,
        )
        print("Training completed successfully.")

        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        return history
