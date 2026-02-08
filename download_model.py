#!/usr/bin/env python3
"""
Helper script to download and convert a plant disease detection model.

This script provides instructions and utilities for obtaining a TFLite model.
"""

import os
import sys

MODEL_INFO = """
========================================
Plant Disease Detection Model Setup
========================================

You need a TensorFlow Lite model file for plant disease detection.

OPTION 1: Download Pre-converted TFLite Model (RECOMMENDED)
------------------------------------------------------------
Visit one of these sources:

1. Kaggle: "Plant Disease Detection TFLite"
   https://www.kaggle.com/models
   Search for: "plant disease mobile" or "plantvillage tflite"

2. GitHub Repositories:
   - Search: "plantvillage tflite model"
   - Look for: plant_disease_model.tflite or similar

3. TensorFlow Hub:
   - Look for MobileNetV2-based plant disease models
   - Convert to TFLite using tensorflow.lite.TFLiteConverter

Download the .tflite file and place it at:
{model_path}

====Model Requirements:====
- Input: 224x224 RGB image
- Output: 38 classes (PlantVillage dataset)
- Format: TFLite (quantized INT8 preferred)
- Size: ~9-15MB

OPT ION 2: Convert from Keras/SavedModel
-----------------------------------------
If you have a Keras .h5 model or SavedModel, you can convert it:

import tensorflow as tf

# Load your model
model = tf.keras.models.load_model('path/to/model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model
with open('plant_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)

OPTION 3: Use Test/Demo Mode
-----------------------------
The application will work in demo mode with test images if no model is found.


For educational purposes, here's a starter Keras model (requires training):
"""

def create_starter_model():
    """
    Creates a basic MobileNetV2 model structure for plant disease detection.
    NOTE: This model is UNTRAINED and for demonstration only!
    """
    try:
        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            ),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(38, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'plant_disease_model.tflite')

    print(MODEL_INFO.format(model_path=model_path))

    print("\nChecking for model file...")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model found: {model_path}")
        print(f"  Size: {size_mb:.2f} MB")
    else:
        print(f"✗ Model not found at: {model_path}")
        print("\nPlease download a model using one of the options above.")

    print("\n" + "="*60)
