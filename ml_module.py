#!/usr/bin/env python3

"""
Machine Learning Module for Plant Disease Detection
====================================================
Handles TensorFlow Lite model loading, image preprocessing,
and inference for plant disease classification.

Educational Project - SunFounder Electronic Kit
"""

import os
import time
import numpy as np
import cv2

class DiseaseDetector:
    """
    Disease Detector class using TensorFlow Lite for inference.
    Loads a quantized model and performs plant disease classification.
    """

    def __init__(self, model_file='models/plant_disease_model.tflite',
                 labels_file='models/labels.txt'):
        """
        Initialize the disease detector.

        Args:
            model_file (str): Path to the TFLite model file
            labels_file (str): Path to the labels text file
        """
        self.model_file = model_file
        self.labels_file = labels_file
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, model_file)
        self.labels_path = os.path.join(self.base_dir, labels_file)

        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = []
        self.model_loaded = False

        # Model configuration
        self.input_size = (300, 300)  # Kaggle model uses 300x300
        self.confidence_threshold = 0.30  # 30% threshold for well-trained model

    def setup(self):
        """
        Load the TFLite model and labels.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try importing TensorFlow Lite
            try:
                import tensorflow as tf
                self.Interpreter = tf.lite.Interpreter
            except ImportError:
                try:
                    import tflite_runtime.interpreter as tflite
                    self.Interpreter = tflite.Interpreter
                except ImportError:
                    print("✗ Error: TensorFlow Lite not found!")
                    print("  Install with: pip3 install tensorflow or tflite-runtime")
                    return False

            # Load labels
            if not os.path.exists(self.labels_path):
                print(f"✗ Error: Labels file not found at {self.labels_path}")
                return False

            with open(self.labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            print(f"✓ Loaded {len(self.labels)} class labels")

            # Load TFLite model
            if not os.path.exists(self.model_path):
                print(f"✗ Error: Model file not found at {self.model_path}")
                print("  See download_model.py for instructions on obtaining a model.")
                return False

            self.interpreter = self.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()

            # Get input and output tensor details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Print model info
            input_shape = self.input_details[0]['shape']
            output_shape = self.output_details[0]['shape']
            model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)

            print(f"✓ Loaded TFLite model ({model_size_mb:.2f} MB)")
            print(f"  Input shape: {input_shape}")
            print(f"  Output shape: {output_shape}")
            print(f"  Expected classes: {len(self.labels)}")

            # Verify output shape matches labels
            if output_shape[-1] != len(self.labels):
                print(f"⚠ Warning: Model output ({output_shape[-1]}) doesn't match labels ({len(self.labels)})")

            self.model_loaded = True

            # Warm up model with dummy prediction
            print("Warming up model...")
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            self.predict(dummy_image)
            print("✓ Model ready for inference")

            return True

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
            return False

    def preprocess_image(self, image):
        """
        Preprocess image for model input.

        Args:
            image (numpy.ndarray): Input image (BGR format from OpenCV)

        Returns:
            numpy.ndarray: Preprocessed image ready for inference
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        image = cv2.resize(image, self.input_size)

        # Check if model expects uint8 or float32 input
        if self.input_details and self.input_details[0]['dtype'] == np.uint8:
            # Model expects uint8 (0-255) - no normalization needed!
            image = image.astype(np.uint8)
        else:
            # Model expects float32 - normalize to [0, 1]
            image = image.astype(np.float32) / 255.0

            # Apply ImageNet normalization (required for ResNet/PyTorch models)
            # mean = [0.485, 0.456, 0.406] for R, G, B channels
            # std = [0.229, 0.224, 0.225] for R, G, B channels
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image = (image - mean) / std

        # Add batch dimension
        image = np.expand_dims(image, axis=0)  # Shape: [1, H, W, 3]

        # Check if model expects channels-first (PyTorch style)
        # Only transpose if channels-first and float32 (not for uint8 models)
        if (self.input_details and
            self.input_details[0]['shape'][1] == 3 and
            self.input_details[0]['dtype'] != np.uint8):
            # Model expects channels-first (PyTorch style): [batch, channels, height, width]
            image = np.transpose(image, (0, 3, 1, 2))  # [1, H, W, 3] -> [1, 3, H, W]

        return image

    def predict(self, image):
        """
        Run inference on an image.

        Args:
            image (numpy.ndarray): Input image (can be any size, will be preprocessed)

        Returns:
            list: List of (class_label, confidence) tuples, sorted by confidence
        """
        if not self.model_loaded:
            print("Error: Model not loaded. Call setup() first.")
            return []

        try:
            # Start timing
            start_time = time.time()

            # Preprocess image
            preprocessed = self.preprocess_image(image)

            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Get predictions
            predictions = output_data[0]

            # Convert uint8 output to float probabilities if needed
            if predictions.dtype == np.uint8:
                predictions = predictions.astype(np.float32) / 255.0

            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            # Create list of (label, confidence) tuples
            results = []
            for i, confidence in enumerate(predictions):
                if i < len(self.labels) and confidence >= self.confidence_threshold:
                    results.append((self.labels[i], float(confidence)))

            # Sort by confidence (highest first)
            results.sort(key=lambda x: x[1], reverse=True)

            print(f"Inference time: {inference_time:.1f}ms")

            return results

        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return []

    def predict_top_n(self, image, n=3):
        """
        Get top N predictions for an image.

        Args:
            image (numpy.ndarray): Input image
            n (int): Number of top predictions to return

        Returns:
            list: Top N (class_label, confidence) tuples
        """
        all_predictions = self.predict(image)
        return all_predictions[:n]

    def format_predictions(self, predictions):
        """
        Format predictions as a human-readable string.

        Args:
            predictions (list): List of (class_label, confidence) tuples

        Returns:
            str: Formatted prediction string
        """
        if not predictions:
            return "No confident predictions (all below threshold)"

        output = []
        output.append("Top Predictions:")
        output.append("-" * 50)

        for idx, (label, confidence) in enumerate(predictions, 1):
            # Format confidence as percentage
            conf_percent = confidence * 100
            # Create confidence bar
            bar_length = int(conf_percent / 5)  # 20 chars max
            bar = "█" * bar_length + "░" * (20 - bar_length)

            output.append(f"{idx}. {label}")
            output.append(f"   {bar} {conf_percent:.1f}%")
            output.append("")

        return "\n".join(output)

    def get_model_info(self):
        """
        Get information about the loaded model.

        Returns:
            dict: Model information
        """
        if not self.model_loaded:
            return {"loaded": False}

        model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)

        return {
            "loaded": True,
            "model_path": self.model_path,
            "model_size_mb": f"{model_size_mb:.2f}",
            "num_classes": len(self.labels),
            "input_size": self.input_size,
            "confidence_threshold": self.confidence_threshold
        }


# Test the ML module
def test_detector():
    """
    Test function for the disease detector.
    """
    print("========================================")
    print("|    Disease Detector Test             |")
    print("========================================\n")

    # Create detector
    detector = DiseaseDetector()

    # Setup (load model)
    if not detector.setup():
        print("Failed to setup detector!")
        return

    # Print model info
    info = detector.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Create a test image (random colored image)
    print("\nTesting with random image...")
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

    # Run prediction
    predictions = detector.predict_top_n(test_image, n=5)

    # Format and print results
    print("\n" + detector.format_predictions(predictions))

    print("\nNote: This is a random test image, so predictions are not meaningful.")
    print("      Use with real plant leaf images for actual disease detection.")


if __name__ == '__main__':
    test_detector()
