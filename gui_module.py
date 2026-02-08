#!/usr/bin/env python3

"""
GUI Module for Plant Disease Detection
=======================================
PyQt5-based graphical user interface that integrates camera, ML model,
and disease database for plant disease detection and diagnosis.

Educational Project - SunFounder Electronic Kit
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QPushButton, QTextEdit,
                            QGroupBox, QStatusBar, QFileDialog)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

class AnalysisThread(QThread):
    """
    Worker thread for running ML inference without blocking the GUI.
    """
    finished = pyqtSignal(object, object)  # Signal: (predictions, inference_time)
    error = pyqtSignal(str)  # Signal: error_message

    def __init__(self, detector, image):
        super().__init__()
        self.detector = detector
        self.image = image

    def run(self):
        try:
            # Run prediction
            predictions = self.detector.predict_top_n(self.image, n=3)
            self.finished.emit(predictions, None)
        except Exception as e:
            self.error.emit(str(e))


class PlantDiseaseGUI(QMainWindow):
    """
    Main GUI window for the Plant Disease Detection System.
    """

    def __init__(self, camera, detector, database):
        super().__init__()
        self.camera = camera
        self.detector = detector
        self.database = database

        self.current_image = None
        self.preview_timer = None
        self.analysis_thread = None

        self.init_ui()
        self.start_preview()

    def init_ui(self):
        """
        Initialize the user interface.
        """
        self.setWindowTitle("Plant Disease Detection System")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel: Camera preview
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=2)

        # Right panel: Results and treatments
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=3)

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

    def create_left_panel(self):
        """
        Create the left panel with camera preview and controls.
        """
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Camera preview label
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout()
        preview_group.setLayout(preview_layout)

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(640, 480)
        self.preview_label.setStyleSheet("border: 2px solid #3498db; background-color: black;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setText("Initializing camera...")
        preview_layout.addWidget(self.preview_label)

        layout.addWidget(preview_group)

        # Control buttons
        control_layout = QVBoxLayout()

        self.capture_button = QPushButton("📸 Capture & Analyze")
        self.capture_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.capture_button.setMinimumHeight(60)
        self.capture_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.capture_button.clicked.connect(self.capture_and_analyze)
        control_layout.addWidget(self.capture_button)

        self.load_button = QPushButton("📁 Load Image File")
        self.load_button.setFont(QFont("Arial", 12))
        self.load_button.setMinimumHeight(50)
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        self.load_button.clicked.connect(self.load_image_file)
        control_layout.addWidget(self.load_button)

        layout.addLayout(control_layout)

        # Add stretch to push everything to top
        layout.addStretch()

        return panel

    def create_right_panel(self):
        """
        Create the right panel with results and treatment information.
        """
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Captured image display
        image_group = QGroupBox("Captured Image")
        image_layout = QVBoxLayout()
        image_group.setLayout(image_layout)

        self.captured_label = QLabel()
        self.captured_label.setFixedSize(400, 300)
        self.captured_label.setStyleSheet("border: 2px solid #e74c3c; background-color: #ecf0f1;")
        self.captured_label.setAlignment(Qt.AlignCenter)
        self.captured_label.setText("No image captured yet")
        image_layout.addWidget(self.captured_label, alignment=Qt.AlignCenter)

        layout.addWidget(image_group)

        # Diagnosis results
        diagnosis_group = QGroupBox("Diagnosis Results")
        diagnosis_layout = QVBoxLayout()
        diagnosis_group.setLayout(diagnosis_layout)

        self.diagnosis_label = QLabel("Waiting for analysis...")
        self.diagnosis_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.diagnosis_label.setStyleSheet("color: #2c3e50; padding: 10px;")
        self.diagnosis_label.setWordWrap(True)
        diagnosis_layout.addWidget(self.diagnosis_label)

        layout.addWidget(diagnosis_group)

        # Treatment recommendations
        treatment_group = QGroupBox("Treatment Recommendations")
        treatment_layout = QVBoxLayout()
        treatment_group.setLayout(treatment_layout)

        self.treatment_text = QTextEdit()
        self.treatment_text.setReadOnly(True)
        self.treatment_text.setFont(QFont("Courier", 10))
        self.treatment_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 10px;
            }
        """)
        self.treatment_text.setPlainText("Capture an image to see treatment recommendations.")
        treatment_layout.addWidget(self.treatment_text)

        layout.addWidget(treatment_group)

        return panel

    def start_preview(self):
        """
        Start the camera preview timer.
        """
        if self.camera and self.camera.camera_available:
            self.preview_timer = QTimer()
            self.preview_timer.timeout.connect(self.update_preview)
            self.preview_timer.start(30)  # Update every 30ms (~33 FPS)
        else:
            self.preview_label.setText("Camera not available\n\nUse 'Load Image File' to analyze images")
            self.preview_label.setStyleSheet("border: 2px solid #e74c3c; background-color: #000000; color: red;")

    def update_preview(self):
        """
        Update the camera preview with a new frame.
        """
        try:
            frame = self.camera.get_preview_frame()
            if frame is not None:
                # Convert RGB to QImage
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

                # Scale to fit preview label
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview_label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"Error updating preview: {e}")

    def capture_and_analyze(self):
        """
        Capture an image and analyze it for plant diseases.
        """
        self.statusBar.showMessage("Capturing image...")
        self.capture_button.setEnabled(False)
        QApplication.processEvents()

        # Capture image
        if self.camera and self.camera.camera_available:
            image = self.camera.capture_image()
        else:
            self.statusBar.showMessage("Camera not available")
            self.capture_button.setEnabled(True)
            return

        if image is None:
            self.statusBar.showMessage("Failed to capture image")
            self.capture_button.setEnabled(True)
            return

        self.current_image = image
        self.display_captured_image(image)

        # Analyze the image
        self.analyze_image(image)

    def load_image_file(self):
        """
        Load an image file from disk for analysis.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Plant Leaf Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            try:
                # Load image using OpenCV
                image = cv2.imread(file_path)
                if image is None:
                    self.statusBar.showMessage("Failed to load image")
                    return

                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                self.current_image = image
                self.display_captured_image(image)
                self.analyze_image(image)

            except Exception as e:
                self.statusBar.showMessage(f"Error loading image: {e}")

    def display_captured_image(self, image):
        """
        Display the captured image in the results panel.
        """
        try:
            # Convert image to QPixmap
            if len(image.shape) == 3:
                height, width, channel = image.shape
                bytes_per_line = 3 * width
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                return

            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.captured_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.captured_label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"Error displaying image: {e}")

    def analyze_image(self, image):
        """
        Analyze the image using ML model in a separate thread.
        """
        self.statusBar.showMessage("Analyzing image...")
        self.diagnosis_label.setText("🔄 Analyzing... Please wait...")
        self.treatment_text.setPlainText("Running inference...")
        QApplication.processEvents()

        # Run analysis in worker thread
        self.analysis_thread = AnalysisThread(self.detector, image)
        self.analysis_thread.finished.connect(self.on_analysis_finished)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.start()

    def on_analysis_finished(self, predictions, _):
        """
        Handle completed analysis.
        """
        self.capture_button.setEnabled(True)

        if not predictions:
            self.diagnosis_label.setText("❌ No confident predictions\n\nTry with better lighting or a clearer leaf image.")
            self.treatment_text.setPlainText("Unable to identify disease with confidence.\n\nTips:\n• Ensure good lighting\n• Focus on a single leaf\n• Capture clear, close-up images\n• Avoid shadows and glare")
            self.statusBar.showMessage("Analysis complete - No confident predictions")
            return

        # Get top prediction
        top_disease, top_confidence = predictions[0]

        # Update diagnosis label
        common_name = self.database.get_common_name(top_disease)
        confidence_percent = top_confidence * 100

        if self.database.is_healthy(top_disease):
            self.diagnosis_label.setText(f"✅ {common_name}\n\nConfidence: {confidence_percent:.1f}%")
            self.diagnosis_label.setStyleSheet("color: #27ae60; padding: 10px; font-size: 14pt;")
        else:
            self.diagnosis_label.setText(f"⚠️  {common_name}\n\nConfidence: {confidence_percent:.1f}%")
            self.diagnosis_label.setStyleSheet("color: #e74c3c; padding: 10px; font-size: 14pt;")

        # Show all top predictions
        predictions_text = "Top 3 Predictions:\n\n"
        for idx, (disease, conf) in enumerate(predictions, 1):
            name = self.database.get_common_name(disease)
            predictions_text += f"{idx}. {name}: {conf*100:.1f}%\n"

        # Get treatment information for top prediction
        treatment_info = self.database.format_treatment_info(top_disease)

        # Display results
        full_text = predictions_text + "\n" + treatment_info
        self.treatment_text.setPlainText(full_text)

        self.statusBar.showMessage(f"Analysis complete - {common_name} detected ({confidence_percent:.1f}% confidence)")

    def on_analysis_error(self, error_msg):
        """
        Handle analysis error.
        """
        self.capture_button.setEnabled(True)
        self.diagnosis_label.setText(f"❌ Analysis Error\n\n{error_msg}")
        self.treatment_text.setPlainText(f"Error during analysis:\n{error_msg}")
        self.statusBar.showMessage("Analysis failed")

    def closeEvent(self, event):
        """
        Handle window close event.
        """
        # Stop preview timer
        if self.preview_timer:
            self.preview_timer.stop()

        # Wait for analysis thread
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.wait()

        event.accept()


# Test the GUI module
def test_gui():
    """
    Test function for the GUI (requires all modules).
    """
    print("========================================")
    print("|    GUI Module Test                   |")
    print("========================================\n")

    # This would normally import and use the actual modules
    # For now, create mock objects

    class MockCamera:
        def __init__(self):
            self.camera_available = False

    class MockDetector:
        def predict_top_n(self, image, n=3):
            return [("Tomato___Early_blight", 0.85), ("Tomato___Late_blight", 0.10), ("Tomato___healthy", 0.05)]

    class MockDatabase:
        def get_common_name(self, label):
            return label.replace("___", " - ").replace("_", " ")

        def is_healthy(self, label):
            return "healthy" in label.lower()

        def format_treatment_info(self, label):
            return f"Mock treatment info for {label}"

    camera = MockCamera()
    detector = MockDetector()
    database = MockDatabase()

    app = QApplication(sys.argv)
    window = PlantDiseaseGUI(camera, detector, database)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    test_gui()
