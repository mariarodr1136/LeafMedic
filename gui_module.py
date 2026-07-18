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
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class AnalysisThread(QThread):
    """
    Worker thread for running ML inference without blocking the GUI.
    """
    finished = pyqtSignal(object, object)  # Signal: (predictions, quality_report)
    error = pyqtSignal(str)  # Signal: error_message

    def __init__(self, detector, image):
        super().__init__()
        self.detector = detector
        self.image = image

    def run(self):
        try:
            # analyze() returns the ranked predictions together with the
            # out-of-distribution and capture-quality signals, so the UI can
            # say "retake this photo" instead of a confident wrong answer.
            report = self.detector.analyze(self.image, n=3)
            self.finished.emit(report["predictions"], report)
        except Exception as e:
            self.error.emit(str(e))


# Palette shared with the web demo (docs/css/style.css)
APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #f4f7f5;
    color: #1c2823;
    font-family: "Helvetica Neue", "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #dce5df;
    border-radius: 10px;
    margin-top: 12px;
    padding: 14px 10px 10px 10px;
    font-weight: 600;
    color: #5b6b62;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    text-transform: uppercase;
}
QTextEdit {
    background-color: #ffffff;
    border: 1px solid #dce5df;
    border-radius: 8px;
    padding: 8px;
}
QStatusBar {
    background-color: #ffffff;
    border-top: 1px solid #dce5df;
    color: #5b6b62;
}
"""


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
        self.setWindowTitle("LeafMedic — Plant Disease Detection")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(APP_STYLESHEET)

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
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

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
        self.preview_label.setStyleSheet("border: 1px solid #dce5df; border-radius: 8px; background-color: #0c110e; color: #9fb3a7;")
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
                background-color: #22996a;
                color: white;
                border-radius: 9px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #1a7a4f;
            }
            QPushButton:pressed {
                background-color: #14603e;
            }
            QPushButton:disabled {
                background-color: #a9b8af;
            }
        """)
        self.capture_button.clicked.connect(self.capture_and_analyze)
        control_layout.addWidget(self.capture_button)

        self.load_button = QPushButton("📁 Load Image File")
        self.load_button.setFont(QFont("Arial", 12))
        self.load_button.setMinimumHeight(50)
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #ffffff;
                color: #1c2823;
                border: 1px solid #dce5df;
                border-radius: 9px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #eef3f0;
            }
            QPushButton:pressed {
                background-color: #dce5df;
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
        self.captured_label.setStyleSheet("border: 1px solid #dce5df; border-radius: 8px; background-color: #eef3f0; color: #5b6b62;")
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
            self.preview_label.setStyleSheet("border: 1px solid #dce5df; border-radius: 8px; background-color: #0c110e; color: #e0a13c;")

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
        self.status_bar.showMessage("Capturing image...")
        self.capture_button.setEnabled(False)
        QApplication.processEvents()

        # Capture image
        if self.camera and self.camera.camera_available:
            image = self.camera.capture_image()
        else:
            self.status_bar.showMessage("Camera not available")
            self.capture_button.setEnabled(True)
            return

        if image is None:
            self.status_bar.showMessage("Failed to capture image")
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
                    self.status_bar.showMessage("Failed to load image")
                    return

                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                self.current_image = image
                self.display_captured_image(image)
                self.analyze_image(image)

            except Exception as e:
                self.status_bar.showMessage(f"Error loading image: {e}")

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
        self.status_bar.showMessage("Analyzing image...")
        self.diagnosis_label.setText("🔄 Analyzing... Please wait...")
        self.treatment_text.setPlainText("Running inference...")
        QApplication.processEvents()

        # Run analysis in worker thread
        self.analysis_thread = AnalysisThread(self.detector, image)
        self.analysis_thread.finished.connect(self.on_analysis_finished)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.start()

    def on_analysis_finished(self, predictions, report=None):
        """
        Handle completed analysis.
        """
        self.capture_button.setEnabled(True)

        if not predictions:
            self.diagnosis_label.setText("❌ No confident predictions\n\nTry with better lighting or a clearer leaf image.")
            self.treatment_text.setPlainText("Unable to identify disease with confidence.\n\nTips:\n• Ensure good lighting\n• Focus on a single leaf\n• Capture clear, close-up images\n• Avoid shadows and glare")
            self.status_bar.showMessage("Analysis complete - No confident predictions")
            return

        # Get top prediction
        top_disease, top_confidence = predictions[0]

        # Update diagnosis label
        common_name = self.database.get_common_name(top_disease)
        confidence_percent = top_confidence * 100

        # An untrustworthy result is presented as Uncertain rather than as a
        # diagnosis: a closed-set softmax is always confident about something.
        trustworthy = report.get("trustworthy", True) if report else True

        if not trustworthy:
            self.diagnosis_label.setText(
                f"❓ Uncertain\n\nClosest match: {common_name} ({confidence_percent:.1f}%)"
            )
            self.diagnosis_label.setStyleSheet("color: #d97706; padding: 10px; font-size: 14pt;")
        elif self.database.is_healthy(top_disease):
            self.diagnosis_label.setText(f"✅ {common_name}\n\nConfidence: {confidence_percent:.1f}%")
            self.diagnosis_label.setStyleSheet("color: #22996a; padding: 10px; font-size: 14pt;")
        else:
            self.diagnosis_label.setText(f"⚠️  {common_name}\n\nConfidence: {confidence_percent:.1f}%")
            self.diagnosis_label.setStyleSheet("color: #dc2626; padding: 10px; font-size: 14pt;")

        self.treatment_text.setHtml(self.build_results_html(predictions, top_disease, report))

        if trustworthy:
            self.status_bar.showMessage(
                f"Analysis complete - {common_name} detected ({confidence_percent:.1f}% confidence)"
            )
        else:
            self.status_bar.showMessage("Analysis complete - result flagged as unreliable")

    def build_quality_html(self, report):
        """Warning banners for out-of-distribution or poor-quality captures."""
        if not report or not report.get("warnings"):
            return ""
        html = []
        for warning in report["warnings"]:
            html.append(
                '<p style="background-color:#fdf3e3; border-left:3px solid #d97706; '
                f'padding:6px 10px; margin:4px 0;">⚠️ {warning}</p>'
            )
        # The raw numbers make the verdict auditable instead of a black box.
        html.append(
            '<p style="color:#5b6b62; font-size:11px;">'
            f'vegetation {report["leaf_score"]:.0%} · '
            f'entropy {report["entropy"]:.2f} · '
            f'sharpness {report["blur_score"]:.0f} · '
            f'brightness {report["mean_luma"]:.0f}'
            '</p>'
        )
        return "".join(html)

    def build_results_html(self, predictions, top_disease, report=None):
        """
        Build rich HTML for the treatment panel: top-3 confidence bars
        followed by the treatment card for the top prediction.
        """
        html = [self.build_quality_html(report)]
        html.append('<h3 style="color:#5b6b62;">Top Predictions</h3>')
        for disease, conf in predictions:
            name = self.database.get_common_name(disease)
            pct = conf * 100
            bar_color = "#22996a" if self.database.is_healthy(disease) else "#d97706"
            html.append(
                f'<div style="margin-bottom:6px;">{name} — <b>{pct:.1f}%</b><br>'
                f'<span style="background-color:{bar_color}; color:{bar_color};">'
                f'{"█" * max(1, int(pct / 4))}</span></div>'
            )

        info = self.database.get_treatment(top_disease)
        if info and not self.database.is_healthy(top_disease):
            sections = [
                ("🔍 Symptoms", info.get('symptoms', [])),
                ("💊 Treatment", info.get('treatments', [])),
                ("🛡️ Prevention", info.get('prevention', [])),
            ]
            html.append(f'<h3 style="color:#5b6b62;">About {info.get("common_name", top_disease)}</h3>')
            html.append(f'<p>{info.get("description", "")}</p>')
            severity = info.get('severity', 'unknown')
            sev_color = {"low": "#22996a", "medium": "#d97706",
                         "high": "#dc2626", "critical": "#991b1b"}.get(severity, "#5b6b62")
            html.append(f'<p>Severity: <b style="color:{sev_color};">{severity.upper()}</b></p>')
            for title, items in sections:
                if items:
                    html.append(f'<h4>{title}</h4><ul>')
                    html.extend(f'<li>{item}</li>' for item in items)
                    html.append('</ul>')
        elif self.database.is_healthy(top_disease):
            html.append('<p style="color:#22996a;"><b>🌿 This leaf looks healthy!</b> '
                        'Keep up regular watering, good airflow, and periodic checks '
                        'of leaf undersides to catch problems early.</p>')

        html.append('<p style="color:#5b6b62;"><i>⚠️ Educational use only — for professional '
                    'diagnosis, consult agricultural extension services.</i></p>')
        return "".join(html)

    def on_analysis_error(self, error_msg):
        """
        Handle analysis error.
        """
        self.capture_button.setEnabled(True)
        self.diagnosis_label.setText(f"❌ Analysis Error\n\n{error_msg}")
        self.treatment_text.setPlainText(f"Error during analysis:\n{error_msg}")
        self.status_bar.showMessage("Analysis failed")

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

        def get_treatment(self, label):
            return {
                "common_name": self.get_common_name(label),
                "description": f"Mock description for {label}",
                "severity": "medium",
                "symptoms": ["Mock symptom"],
                "treatments": ["Mock treatment"],
                "prevention": ["Mock prevention"],
            }

    camera = MockCamera()
    detector = MockDetector()
    database = MockDatabase()

    app = QApplication(sys.argv)
    window = PlantDiseaseGUI(camera, detector, database)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    test_gui()
