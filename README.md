# LeafMedic: Embedded Computer Vision & Machine Learning System 🪴🔎

**LeafMedic: Embedded Computer Vision & Machine Learning System** is an end-to-end **edge AI application** that performs real-time plant disease classification using **computer vision** and **machine learning**, fully deployed on a **Raspberry Pi**. Built with **Python**, the system integrates camera-based image acquisition, an optimized **TensorFlow Lite** **MobileNet** model for on-device inference, and a modular software architecture designed for performance, scalability, and maintainability.

The application captures high-resolution leaf images, dynamically preprocesses them to match model requirements, and executes low-latency inference directly on embedded hardware. A responsive **PyQt5** graphical interface presents confidence-ranked predictions along with structured treatment recommendations sourced from a curated disease knowledge base.

This project demonstrates practical, production-oriented engineering skills across **machine learning deployment**, **software architecture**, and **edge computing**, highlighting the ability to move beyond experimental notebooks and deliver robust, real-world AI systems. It reflects hands-on experience with **ML model integration**, **hardware–software interaction**, **performance optimization**, and **user-focused application design**.

---

![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4B-red) ![Python](https://img.shields.io/badge/Python-3.7%2B-blue) ![TensorFlow Lite](https://img.shields.io/badge/TensorFlow-Lite-orange) ![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV-green) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Edge%20AI-purple) ![GUI](https://img.shields.io/badge/GUI-PyQt5-brightgreen) ![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey) ![Model](https://img.shields.io/badge/Model-MobileNetV1-blueviolet) ![Inference](https://img.shields.io/badge/Inference-Real--Time-success) ![License](https://img.shields.io/badge/License-Educational-yellowgreen)

---

## 📋 Table of Contents

- [Features](#-features)
- [Hardware Requirements](#-hardware-requirements)
- [Software Dependencies](#-software-dependencies)
- [Installation](#-installation)
- [Usage](#-usage)
- [Supported Plants & Diseases](#-supported-plants--diseases)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Educational Purpose](#-educational-purpose)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)


---

## ✨ Features

- **Real-time Camera Preview**: Live preview from Raspberry Pi Camera Module
- **Automated Disease Detection**: ML-powered identification of 16 plant disease classes (90%+ accuracy)
- **Treatment Recommendations**: Detailed treatment and prevention information for 43 diseases
- **User-Friendly GUI**: PyQt5-based graphical interface
- **Batch Processing**: Analyze images from files
- **Educational**: Learn about plant diseases, computer vision, and edge AI deployment
- **Fast Inference**: ~145ms per image on Raspberry Pi 4

---

## 🔧 Hardware Requirements

- **Raspberry Pi 4 Model B** (4GB recommended)
- **Camera**: Arducam 5MP OV5647 Camera Module V1 (or compatible)
- **Display**: Monitor/screen for GUI (1200x800 minimum recommended)
- **Storage**: 2GB free space for model and dependencies
- **Power**: 5V 3A USB-C power supply

### Camera Connection

1. Power off Raspberry Pi
2. Locate camera connector (between HDMI ports and audio jack)
3. Pull up plastic clip gently
4. Insert ribbon cable (**blue side facing audio jack**, contacts facing HDMI)
5. Push clip down to secure
6. Power on Raspberry Pi

---

## 📦 Software Dependencies

### System Requirements

- **OS**: Raspberry Pi OS (64-bit recommended)
- **Python**: 3.7 or higher
- **Camera**: libcamera enabled

### Python Libraries

- `tensorflow` (2.x) - Machine learning framework
- `PyQt5` - GUI framework
- `opencv-python` - Image processing
- `picamera2` - Camera interface
- `numpy` - Numerical operations

---

## 🚀 Installation

### Step 1: Enable Camera

```bash
# Method 1: Using raspi-config
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable

# Method 2: Edit config (already done if camera works)
sudo nano /boot/firmware/config.txt
# Ensure: camera_auto_detect=1
```

### Step 2: Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
pip3 install tensorflow opencv-python --break-system-packages

# Verify Picamera2 and PyQt5 (usually pre-installed)
python3 -c "import picamera2; import PyQt5; print('✓ Libraries OK')"
```

### Step 3: Download Project

```bash
cd ~/electronic-kit/for-raspberry-pi/python/21_PlantDiseaseDetection
```

### Step 4: Obtain ML Model

**Current Model**: The project includes the **Kaggle AgriPredict Disease Classification model** (12MB TFLite, trained and production-ready).

- **Source**: [Kaggle AgriPredict Disease Classification](https://www.kaggle.com/models/agripredict/disease-classification)
- **Architecture**: MobileNetV1 (optimized for edge devices)
- **Classes**: 16 disease categories across 4 crops
- **Accuracy**: 90%+ confidence on well-captured images
- **Inference**: ~145ms on Raspberry Pi 4
- **Input**: 300×300 RGB images (uint8, no normalization needed)

The model file is located at:
```
models/plant_disease_model.tflite
```

#### Future Expansion

Plan to expand disease coverage by:
1. Training on PlantVillage dataset (54,000+ images, 38 classes)
2. Adding support for more crops (Apple, Grape, Potato, Strawberry, etc.)
3. Creating ensemble models for improved accuracy
4. Supporting multi-disease detection per image

---

## 🎮 Usage

### Running the Application

```bash
cd ~/electronic-kit/for-raspberry-pi/python/21_PlantDiseaseDetection
python3 main.py
```

### GUI Instructions

1. **Live Preview**: Camera preview appears in left panel
2. **Capture & Analyze**: Click button to capture and analyze leaf
3. **View Results**: See diagnosis, confidence, and treatment recommendations
4. **Load Image**: Analyze saved images using "Load Image File" button

### Optimal Image Capture Tips

- **Distance**: 20-30 cm from leaf
- **Lighting**: Natural daylight or bright LED (avoid shadows)
- **Focus**: Single leaf, fill most of frame
- **Angle**: Perpendicular to leaf surface
- **Background**: Plain, contrasting background helps

### Command-Line Testing

Test individual modules:

```bash
# Test camera
python3 camera_module.py

# Test ML model
python3 ml_module.py

# Test disease database
python3 disease_database.py
```

---

## 🌿 Supported Plants & Diseases

### Currently Supported (16 Classes - Kaggle AgriPredict Model)

The system **currently detects** the following plant diseases with **90%+ accuracy**:

#### 🍅 Tomato (8 classes)
- ✓ **Healthy** - No disease present
- ⚠️ **Bacterial Spot** - Dark spots with yellow halos (High severity)
- ⚠️ **Septoria Leaf Spot** - Small circular spots with gray centers (High severity)
- ⚠️ **Late Blight** - Devastating disease, can destroy crops rapidly (Critical severity)
- ⚠️ **Leaf Mold** - Yellow spots with fuzzy growth underneath (Medium severity)
- ⚠️ **Spider Mites** - Pest damage causing yellow stippling (Medium severity)
- ⚠️ **Yellow Leaf Curl Virus (TYLCV)** - Viral disease spread by whiteflies (Critical severity)
- *(1 additional class in model)*

#### 🌽 Corn/Maize (4 classes)
- ✓ **Healthy** - No disease present
- ⚠️ **Common Rust** - Reddish-brown pustules on leaves (Medium severity)
- ⚠️ **Gray Leaf Spot (Cercospora)** - Long narrow gray lesions (High severity)
- ⚠️ **Lethal Necrosis (MLN)** - Devastating viral disease, no cure (Critical severity)

#### 🌱 Soybean (3 classes)
- ✓ **Healthy** - No disease present
- ⚠️ **Frogeye Leaf Spot** - Circular lesions with gray centers (Medium severity)
- ⚠️ **Downy Mildew** - Yellow spots with fuzzy growth (Medium severity)

#### 🥬 Cabbage (2 classes)
- ✓ **Healthy** - No disease present
- ⚠️ **Black Rot** - V-shaped yellow lesions, bacterial disease (High severity)

**Total**: 16 disease classes across **4 crop types**

---

### Future Planned Additions

Prepared treatment information for **43 total diseases** in the database. Future model versions will support:

#### 🍎 Apple (4 classes)
- Apple Scab, Black Rot, Cedar Apple Rust, Healthy

#### 🫐 Blueberry (1 class)
- Healthy

#### 🍒 Cherry (2 classes)
- Powdery Mildew, Healthy

#### 🍇 Grape (4 classes)
- Black Rot, Esca (Black Measles), Leaf Blight, Healthy

#### 🍊 Orange/Citrus (1 class)
- Huanglongbing (Citrus Greening) - Critical

#### 🍑 Peach (2 classes)
- Bacterial Spot, Healthy

#### 🌶️ Bell Pepper (2 classes)
- Bacterial Spot, Healthy

#### 🥔 Potato (3 classes)
- Early Blight, Late Blight, Healthy

#### 🍓 Strawberry (2 classes)
- Leaf Scorch, Healthy

#### 🥒 Squash (1 class)
- Powdery Mildew

#### 🫛 Raspberry (1 class)
- Healthy

Plus additional Tomato diseases (Early Blight, Target Spot, Mosaic Virus) and expanded Corn variants.

---

## 🧠 How It Works

### System Architecture

```
┌─────────────┐      ┌──────────────┐      ┌────────────────┐
│   Camera    │─────▶│ Preprocessing│─────▶│   TFLite Model │
│  (OV5647)   │      │ (300x300 RGB)│      │  (MobileNetV1) │
└─────────────┘      └──────────────┘      └────────┬───────┘
                                                     │
                                                     ▼
┌─────────────┐      ┌──────────────┐      ┌────────────────┐
│     GUI     │◀─────│   Treatment  │◀─────│  Predictions   │
│   Display   │      │   Database   │      │  (Top 3 + %)   │
└─────────────┘      └──────────────┘      └────────────────┘
```

### Processing Pipeline

1. **Capture**: Camera captures 2592x1944 RGB image
2. **Preprocessing**:
   - Resize to 300x300 pixels (Kaggle model requirement)
   - Convert BGR to RGB color space
   - Keep uint8 format (0-255) - **no normalization needed**
   - No ImageNet normalization (model uses raw pixel values)
3. **Inference**:
   - TensorFlow Lite model processes image
   - Outputs probabilities for 16 classes
   - Takes ~145ms on Raspberry Pi 4
4. **Post-processing**:
   - Convert uint8 output to float probabilities
   - Filter predictions above confidence threshold (30%)
   - Sort by confidence (highest first)
   - Return top 3 predictions
5. **Display**:
   - Show disease name and confidence percentage
   - Retrieve treatment information from JSON database
   - Display symptoms, treatments, and prevention methods

### Model Details

**Current Production Model** (Kaggle AgriPredict):
- **Architecture**: MobileNetV1 (lightweight CNN optimized for mobile/edge)
- **Input**: 300×300×3 RGB image (uint8, range 0-255)
- **Output**: 16 classes with softmax probabilities (uint8, converted to 0-1)
- **Size**: 12MB (unquantized)
- **Dataset**: AgriPredict internal dataset (proprietary)
- **Training**: Pre-trained on crops: Tomato, Corn, Soybean, Cabbage
- **Inference Speed**: ~145ms on Raspberry Pi 4 (4GB)
- **Accuracy**: 90%+ confidence on well-captured leaf images

**Model Development Journey**:

Tested multiple models during development:

1. **Demo Model** (Initial): 2.6MB untrained model for testing infrastructure
   - ❌ Poor accuracy (<5% confidence)
   - ✓ Good for validating system architecture

2. **PlantAi GitHub Model**: 11MB model with PyTorch-style preprocessing
   - ❌ Required channels-first format [1,3,H,W]
   - ❌ Required ImageNet normalization (mean/std)
   - ❌ Only 6-7% confidence (undertrained)
   - ✓ Demonstrated different preprocessing requirements

3. **Kaggle AgriPredict Model** (Final): 12MB production model
   - ✓ **90% confidence** - well-trained!
   - ✓ Simple preprocessing (no normalization)
   - ✓ Channels-last format [1,H,W,3] (TensorFlow standard)
   - ✓ Fast inference (~145ms)
   - ✓ Proven accuracy in production

**Key Technical Lessons**:

ML module (`ml_module.py`) now intelligently handles:
- **Automatic format detection**: Checks if model expects uint8 or float32
- **Channels-first vs channels-last**: Auto-transposes if needed [1,H,W,3] ↔ [1,3,H,W]
- **ImageNet normalization**: Applies mean/std normalization only for PyTorch models
- **Dynamic input sizing**: Supports 224×224, 300×300, or any model input size
- **Output conversion**: Handles uint8 outputs (0-255 → 0-1 probabilities)

---

## 📁 Project Structure

```
21_PlantDiseaseDetection/
├── main.py                      # Main application (run this!)
├── camera_module.py             # Camera control with Picamera2
├── ml_module.py                 # ML inference with TensorFlow Lite
├── gui_module.py                # PyQt5 graphical interface
├── disease_database.py          # Disease information management
├── download_model.py            # Helper script for model setup (unused)
├── models/
│   ├── plant_disease_model.tflite  # AgriPredict model (12MB, 16 classes)
│   └── labels.txt               # 16 class labels (Kaggle order)
├── data/
│   └── treatments.json          # Disease treatment database (43 diseases)
├── test_images/                 # Sample test images (Tomato, Corn, Soybean only)
└── README.md                    # This file
```

### Module Descriptions

- **main.py**: Entry point, coordinates all modules
- **camera_module.py**: `CameraController` class for camera operations
- **ml_module.py**: `DiseaseDetector` class for ML inference
- **gui_module.py**: `PlantDiseaseGUI` class for user interface
- **disease_database.py**: `TreatmentDatabase` class for treatment info

---

## 🔍 Troubleshooting

### Camera Issues

**Problem**: "Camera not available"
```bash
# Check camera is detected
libcamera-hello --list-cameras

# Should show: ov5647 [2592x1944]

# If not detected:
sudo raspi-config  # Enable camera
sudo reboot
```

**Problem**: "Camera in use by another process"
```bash
# Find and kill process
sudo pkill -9 libcamera
sudo pkill -9 rpicam
```

### ML Model Issues

**Problem**: "Model file not found"
- Ensure `models/plant_disease_model.tflite` exists (should be 12MB Kaggle model)
- The model is included in the project repository

**Problem**: "No module named 'tensorflow'"
```bash
pip3 install tensorflow --break-system-packages
# Note: May need to upgrade flatbuffers if you get import errors
pip3 install --upgrade flatbuffers --break-system-packages
```

**Problem**: Slow inference (>300ms)
- Current model runs at ~145ms on Pi 4, which is optimal
- Close other applications to free memory if slower
- Ensure you're using the Kaggle AgriPredict model (not an older demo model)

**Problem**: Low confidence predictions (<50%)
- **Good news**: Current model achieves 90%+ on proper images!
- Ensure you're analyzing the correct plant types (Tomato, Corn, Soybean, Cabbage only)
- See "Optimal Image Capture Tips" above for best results

### Display Issues

**Problem**: GUI doesn't appear
```bash
# Ensure DISPLAY is set (if SSH)
export DISPLAY=:0

# Or run directly on Pi with monitor
```

**Problem**: "cannot connect to X server"
- Must run on Pi desktop, not headless SSH
- Or use VNC with desktop forwarding

### Low Confidence Predictions

**If the current model (90%+ accuracy) gives low confidence**:
- Ensure good lighting (natural daylight is best)
- Capture close-up of single leaf filling most of the frame
- Check leaf has clear disease symptoms (not just aged/dying leaves)
- Verify you're analyzing supported crops: **Tomato, Corn, Soybean, or Cabbage only**
- Try different angle or distance (20-30cm optimal)
- Avoid blurry images - hold camera steady

**If analyzing unsupported plants** (Apple, Grape, Potato, etc.):
- Model will give random low-confidence predictions
- Expansion to more crops planned for future versions
- See "Future Planned Additions" section above

### Memory Issues

**Problem**: "Out of memory" errors
```bash
# Check available memory
free -h

# Close other applications
# Use lighter model (INT8 quantized)
```

---

## 📚 Educational Purpose

This project is designed for **educational purposes** to demonstrate:

- Raspberry Pi hardware integration
- Computer vision with OpenCV
- Machine learning with TensorFlow Lite
- GUI development with PyQt5
- Modular Python programming
- Edge AI deployment

### Learning Objectives

1. **Hardware**: Camera interfacing, GPIO basics
2. **Software**: Python OOP, threading, event handling
3. **AI/ML**: Image preprocessing, model inference, confidence thresholds
4. **Data**: JSON databases, structured information retrieval
5. **UX**: GUI design, user feedback, error handling

### Limitations & Disclaimers

⚠️ **IMPORTANT**:

- This is a **learning project**, not a professional diagnostic tool
- Current model limited to **4 crops** (Tomato, Corn, Soybean, Cabbage) with **16 disease classes**
- Predictions achieve 90%+ accuracy on well-captured images, but:
  - Should NOT replace expert agricultural advice
  - False positives/negatives can occur
  - Some diseases look similar and may be confused
- For real crop management decisions, consult agricultural extension services or plant pathologists
- Works best with clear, well-lit leaf images showing obvious symptoms
- Cannot detect multiple diseases on single leaf (picks highest confidence)
- Only detects diseases in training set (won't identify novel diseases)

---

### Model Development Journey

**Phase 1: Demo Model (2.6MB)**
- Purpose: Test system architecture
- Result: <5% confidence (untrained model)
- Lesson: Infrastructure worked, but model quality is crucial

**Phase 2: PlantAi GitHub Model (11MB)**
- Challenges:
  - Required channels-first format: [1, 3, 224, 224] (PyTorch style)
  - Needed ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  - Dimension mismatch errors initially
- Solutions:
  - Added automatic transposition: [1,H,W,3] → [1,3,H,W]
  - Implemented ImageNet normalization
- Result: Only 6-7% confidence (model undertrained)
- Lesson: Preprocessing must match training exactly

**Phase 3: Kaggle AgriPredict Model (12MB) - PRODUCTION**
- Specifications:
  - Input: 300×300 uint8 (no normalization!)
  - Output: uint8 probabilities (need /255.0 conversion)
  - Channels-last format (TensorFlow standard)
- Result: **90.2% confidence** - SUCCESS!
- Lesson: Production-quality trained models make all the difference

### Preprocessing Pipeline Evolution

The `ml_module.py` evolved to handle multiple model formats:

```python
# Key features added:
1. Automatic dtype detection (uint8 vs float32)
2. Conditional ImageNet normalization
3. Dynamic channels transposition
4. Flexible input sizing (224x224, 300x300, etc.)
5. Output format conversion
```

### Database Expansion

**treatments.json Evolution**:
- Started with 38 PlantVillage diseases
- Expanded to 43 diseases to include:
  - Soybean Frogeye Leaf Spot
  - Soybean Downy Mildew
  - Corn Lethal Necrosis (Critical severity)
  - Cabbage Healthy
  - Cabbage Black Rot

Each entry includes:
- Common name and scientific name
- Detailed symptoms list
- Treatment recommendations
- Prevention strategies
- Severity rating (none, medium, high, critical)

### Performance Optimization

**Inference Speed**:
- Current: ~145ms per image
- Well within acceptable range for interactive use
- Camera warm-up: 2 seconds on startup
- GUI stays responsive (QThread for non-blocking inference)

**Memory Usage**:
- Model: 12MB
- TensorFlow runtime: ~200-300MB
- GUI application: ~100-150MB
- Camera buffers: ~50MB
- **Total**: ~400-500MB (well within Pi 4 4GB capacity)

### Key Technical Lessons

1. **Always verify model preprocessing requirements**
   - Check: uint8 vs float32
   - Check: Channels-first vs channels-last
   - Check: Normalization method (raw, [0-1], ImageNet)
   - Check: Input dimensions

2. **Model quality matters more than code optimization**
   - 90% trained model >>> perfectly optimized preprocessing pipeline for 5% model

3. **Label order must match training exactly**
   - Even with correct preprocessing, wrong labels = wrong diagnosis
   - Always verify against model documentation

4. **Test with real data early**
   - Demo models don't reveal preprocessing bugs
   - Real confidence scores reveal training quality

5. **Modular design enables experimentation**
   - Separate camera, ML, GUI, database modules
   - Easy to swap models and test different approaches
   - Each module testable independently

---

## 🤝 Contributing

This is an educational project. Contributions welcome:

- Improved models with higher accuracy
- Additional plant species and diseases
- Better treatment recommendations
- GUI improvements
- Bug fixes and optimizations

---

## 📄 License

**Educational Use Only**

This project is part of the SunFounder Raspberry Pi Electronic Kit and is intended for educational purposes. The code is provided as-is for learning computer vision and machine learning concepts.

**Model & Data**:
- Kaggle AgriPredict: Disease Classification model
- AgriPredict internal dataset (proprietary, 16 classes)
- PlantVillage dataset: CC0 (Public Domain) - for future expansion
- Treatment information: Compiled from public agricultural resources

**Dependencies**:
- TensorFlow: Apache 2.0 License
- PyQt5: GPL v3
- OpenCV: Apache 2.0 License

---

## 👏 Acknowledgments

- **AgriPredict**: Disease Classification TFLite model on Kaggle
- **PlantVillage**: Dataset for future model training (54,000+ images)
- **SunFounder**: Electronic kit and educational platform
- **TensorFlow**: ML framework and TFLite runtime
- **Raspberry Pi Foundation**: Hardware platform and Picamera2 library
- **PyQt5**: GUI framework for desktop applications


---

## 🌐 Contact
If you have any questions or feedback, feel free to reach out at [mrodr.contact@gmail.com](mailto:mrodr.contact@gmail.com).

*Educational Project - Learn, Experiment, Innovate*
