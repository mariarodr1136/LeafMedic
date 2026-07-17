#!/usr/bin/env python3

"""
LeafMedic — Plant Disease Detection
====================================
Main application that integrates camera, ML model, and disease database
to identify plant diseases and provide treatment recommendations.

Runs on a Raspberry Pi with a camera module, or on any desktop/laptop
with a webcam. Prefer no install at all? Use the browser demo:
https://mariarodr1136.github.io/LeafMedic/
"""

import sys
from PyQt5.QtWidgets import QApplication

# Import custom modules
from camera_module import CameraController
from ml_module import DiseaseDetector
from disease_database import TreatmentDatabase
from gui_module import PlantDiseaseGUI

# Global references for cleanup
camera = None
detector = None
database = None
app = None


def print_message():
    """
    Print startup message with project information.
    """
    print("=" * 60)
    print("  LeafMedic — Plant Disease Detection")
    print("  MobileNetV2 TFLite · 16 conditions · 4 crops")
    print("  Camera: Raspberry Pi module or any webcam")
    print("  Educational project — not professional advice")
    print("=" * 60)
    print()
    print("Starting up...")
    print()


def setup():
    """
    Initialize all components: camera, ML model, and disease database.

    Returns:
        bool: True if all components initialized successfully
    """
    global camera, detector, database

    print("=" * 60)
    print("SETUP PHASE - Initializing Components")
    print("=" * 60)
    print()

    # Initialize camera controller
    print("[1/3] Initializing Camera...")
    print("-" * 60)
    camera = CameraController(preview_size=(640, 480), capture_size=(2592, 1944))
    if not camera.setup():
        print("⚠ Warning: Camera initialization failed!")
        print("  You can still use the application with image files.")
        print()
    print()

    # Initialize ML detector
    print("[2/3] Loading ML Model...")
    print("-" * 60)
    detector = DiseaseDetector(
        model_file='models/plant_disease_model.tflite',
        labels_file='models/labels.txt'
    )
    if not detector.setup():
        print("✗ Error: Failed to load ML model!")
        print("  The system cannot function without a model.")
        return False
    print()

    # Load disease database
    print("[3/3] Loading Disease Treatment Database...")
    print("-" * 60)
    database = TreatmentDatabase(data_file='data/treatments.json')
    if not database.load():
        print("✗ Error: Failed to load disease database!")
        print("  The system cannot provide treatment recommendations.")
        return False
    print()

    print("=" * 60)
    print("✓ ALL COMPONENTS INITIALIZED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Supported Plants:")
    print("  • Tomato, Corn (maize), Soybean, Cabbage")
    print()
    print("Instructions:")
    print("  1. Point camera at a plant leaf")
    print("  2. Click 'Capture & Analyze' button")
    print("  3. View diagnosis and treatment recommendations")
    print("  4. OR use 'Load Image File' to analyze saved images")
    print()
    print("Note: This system is for EDUCATIONAL purposes only.")
    print("      For professional diagnosis, consult agricultural experts.")
    print()
    print("=" * 60)
    print()

    return True


def main():
    """
    Main function - Launch the GUI application.
    """
    global app

    # Print startup message
    print_message()

    # Initialize components
    if not setup():
        print("\n✗ Setup failed. Cannot start the application.")
        print("Please check the error messages above and fix the issues.")
        return

    print("Launching GUI application...")
    print("Please press Ctrl+C in terminal or close the window to exit.")
    print()

    # Create PyQt5 application
    app = QApplication(sys.argv)
    app.setApplicationName("Plant Disease Detection System")

    # Create main window with all components
    window = PlantDiseaseGUI(camera, detector, database)
    window.show()

    # Run application event loop
    exit_code = app.exec_()

    # Cleanup after GUI closes
    destroy()

    sys.exit(exit_code)


def destroy():
    """
    Cleanup function - Release all resources.
    """
    print()
    print("=" * 60)
    print("CLEANUP - Releasing Resources")
    print("=" * 60)

    # Cleanup camera
    if camera is not None:
        print("Releasing camera...")
        camera.destroy()

    # Other modules don't need explicit cleanup

    print("✓ All resources released successfully")
    print()
    print("Thank you for using the Plant Disease Detection System!")
    print("=" * 60)


# Entry point
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\n\nKeyboard interrupt detected (Ctrl+C)")
        destroy()
    except Exception as e:
        # Handle unexpected errors
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        destroy()
