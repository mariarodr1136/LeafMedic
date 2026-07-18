#!/usr/bin/env python3

"""
Camera Module for Plant Disease Detection
==========================================
Handles camera operations for LeafMedic. Prefers the Raspberry Pi camera
(Picamera2) when available and falls back to any regular webcam via OpenCV,
so the app runs on laptops and desktops too.
"""

import time
from typing import Any

import numpy as np


class CameraController:
    """
    Camera Controller supporting two backends:
      - 'picamera2': Raspberry Pi camera module
      - 'opencv':    any webcam via cv2.VideoCapture
    Both expose the same interface: get_preview_frame() and capture_image()
    return RGB numpy arrays.
    """

    def __init__(self, preview_size=(640, 480), capture_size=(2592, 1944)):
        """
        Initialize the camera controller.

        Args:
            preview_size (tuple): Size for preview frames (width, height)
            capture_size (tuple): Size for captured images (width, height)
        """
        self.preview_size = preview_size
        self.capture_size = capture_size
        # Typed as Any: Picamera2 / cv2.VideoCapture are optional imports.
        self.camera: Any = None
        self.webcam: Any = None
        self.backend = None
        self.camera_available = False
        self.preview_config = None
        self.capture_config = None

    def setup(self):
        """
        Initialize and configure the camera. Tries the Raspberry Pi camera
        first, then falls back to a regular webcam.

        Returns:
            bool: True if successful, False otherwise
        """
        if self._setup_picamera2():
            return True
        return self._setup_webcam()

    def _setup_picamera2(self):
        """Try to initialize the Raspberry Pi camera via Picamera2."""
        try:
            try:
                from picamera2 import Picamera2
            except ImportError:
                return False

            # Create camera instance
            self.camera = Picamera2()

            # Create preview configuration (lower resolution for speed)
            self.preview_config = self.camera.create_preview_configuration(
                main={"size": self.preview_size, "format": "RGB888"}
            )

            # Create capture configuration (high resolution)
            self.capture_config = self.camera.create_still_configuration(
                main={"size": self.capture_size, "format": "RGB888"}
            )

            # Start with preview configuration
            self.camera.configure(self.preview_config)
            self.camera.start()

            # Give camera time to warm up
            time.sleep(2)

            self.backend = 'picamera2'
            self.camera_available = True
            print("✓ Raspberry Pi camera initialized (Picamera2)")
            print(f"  Preview size: {self.preview_size}")
            print(f"  Capture size: {self.capture_size}")

            return True

        except Exception as e:
            print(f"  Picamera2 unavailable ({e}), trying webcam...")
            self.camera = None
            return False

    def _setup_webcam(self):
        """Fall back to a regular webcam via OpenCV."""
        try:
            import cv2
            self.webcam = cv2.VideoCapture(0)
            if not self.webcam.isOpened():
                print("✗ Error: No camera found (neither Pi camera nor webcam)")
                print("  You can still use the application with image files.")
                self.webcam = None
                return False

            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.preview_size[0])
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.preview_size[1])
            # Grab one frame to confirm the device actually delivers images
            ok, _ = self.webcam.read()
            if not ok:
                print("✗ Error: Webcam opened but returned no frames")
                self.webcam.release()
                self.webcam = None
                return False

            self.backend = 'opencv'
            self.camera_available = True
            actual = (int(self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print(f"✓ Webcam initialized (OpenCV, {actual[0]}x{actual[1]})")
            return True

        except Exception as e:
            print(f"✗ Error initializing webcam: {e}")
            self.camera_available = False
            return False

    def get_preview_frame(self):
        """
        Get a single preview frame from the camera.

        Returns:
            numpy.ndarray: Preview frame in RGB format, or None if unavailable
        """
        if not self.camera_available:
            print("Warning: Camera not available")
            # Return a blank frame with "No Camera" message
            blank = np.zeros((self.preview_size[1], self.preview_size[0], 3), dtype=np.uint8)
            return blank

        try:
            if self.backend == 'picamera2':
                # Capture array (returns numpy array)
                return self.camera.capture_array()

            import cv2
            ok, frame = self.webcam.read()
            if not ok:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(f"Error capturing preview frame: {e}")
            return None

    def capture_image(self):
        """
        Capture a high-resolution image for analysis.

        Returns:
            numpy.ndarray: Captured image in RGB format, or None if unavailable
        """
        if not self.camera_available:
            print("Error: Camera not available for capture")
            return None

        if self.backend == 'opencv':
            # Webcams deliver one resolution; a preview frame is the capture.
            image = self.get_preview_frame()
            if image is not None:
                print(f"✓ Captured image: {image.shape}")
            return image

        try:
            # Switch to capture configuration
            self.camera.stop()
            self.camera.configure(self.capture_config)
            self.camera.start()

            # Allow camera to adjust
            time.sleep(0.5)

            # Capture the image
            image = self.camera.capture_array()

            # Switch back to preview configuration
            self.camera.stop()
            self.camera.configure(self.preview_config)
            self.camera.start()

            print(f"✓ Captured image: {image.shape}")
            return image

        except Exception as e:
            print(f"Error capturing image: {e}")
            # Try to recover by restarting preview
            try:
                self.camera.stop()
                self.camera.configure(self.preview_config)
                self.camera.start()
            except Exception:
                pass
            return None

    def save_image(self, image, filename):
        """
        Save an image to disk.

        Args:
            image (numpy.ndarray): Image to save
            filename (str): Output filename

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import cv2
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, image_bgr)
            print(f"✓ Image saved: {filename}")
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False

    def get_camera_info(self):
        """
        Get information about the camera.

        Returns:
            dict: Camera information
        """
        if not self.camera_available:
            return {
                "available": False,
                "message": "Camera not initialized"
            }

        if self.backend == 'opencv':
            return {
                "available": True,
                "model": "Webcam (OpenCV)",
                "preview_size": self.preview_size,
                "capture_size": self.preview_size
            }

        try:
            camera_properties = self.camera.camera_properties
            return {
                "available": True,
                "model": camera_properties.get('Model', 'Unknown'),
                "preview_size": self.preview_size,
                "capture_size": self.capture_size
            }
        except Exception as e:
            return {
                "available": True,
                "error": str(e)
            }

    def destroy(self):
        """
        Clean up camera resources.
        """
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.close()
                print("✓ Camera resources released")
            except Exception as e:
                print(f"Error releasing camera: {e}")
        if self.webcam is not None:
            try:
                self.webcam.release()
                print("✓ Webcam released")
            except Exception as e:
                print(f"Error releasing webcam: {e}")
            self.webcam = None
        self.camera_available = False


# Test the camera module
def test_camera():
    """
    Test function for the camera controller.
    """
    print("========================================")
    print("|    Camera Controller Test            |")
    print("========================================\n")

    # Create camera controller
    camera = CameraController()

    # Setup camera
    if not camera.setup():
        print("Failed to setup camera!")
        print("\nTesting with simulated camera...")
        test_simulated_camera()
        return

    # Print camera info
    info = camera.get_camera_info()
    print("\nCamera Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    try:
        # Test preview frames
        print("\nTesting preview frames (5 frames)...")
        for i in range(5):
            frame = camera.get_preview_frame()
            if frame is not None:
                print(f"  Frame {i+1}: shape={frame.shape}, dtype={frame.dtype}")
            time.sleep(0.5)

        # Test high-resolution capture
        print("\nTesting high-resolution capture...")
        image = camera.capture_image()
        if image is not None:
            print(f"  Captured: shape={image.shape}, dtype={image.dtype}")

            # Save the image
            test_filename = "test_capture.jpg"
            if camera.save_image(image, test_filename):
                print(f"  Test image saved as: {test_filename}")

    finally:
        # Cleanup
        print("\nCleaning up...")
        camera.destroy()
        print("Camera test complete!")


def test_simulated_camera():
    """
    Test with a simulated camera (no hardware required).
    """
    import cv2

    print("\n--- Simulated Camera Mode ---")
    print("This demonstrates the camera module without actual hardware.")

    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some color
    test_image[:160, :] = [100, 50, 50]  # Top third - red
    test_image[160:320, :] = [50, 100, 50]  # Middle - green
    test_image[320:, :] = [50, 50, 100]  # Bottom - blue

    # Add text
    cv2.putText(test_image, "Simulated Camera", (150, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    print(f"Simulated frame shape: {test_image.shape}")

    # Save test image
    filename = "simulated_camera_test.jpg"
    cv2.imwrite(filename, test_image)
    print(f"Simulated image saved as: {filename}")


if __name__ == '__main__':
    test_camera()
