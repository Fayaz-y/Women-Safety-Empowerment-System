# Women Safety Empowerment System

## Project Overview

The Women Safety Empowerment System is an AI-powered surveillance solution designed to enhance women's safety in public and private spaces. This system uses computer vision (CV) and artificial intelligence to monitor camera feeds in real-time, detecting suspicious activities, violent behavior, and potential threats to women. When danger is detected, the system automatically sends alerts to nearby police stations and authorities, enabling rapid response to emergency situations.

## Features

- **Real-Time Violence Detection**: Monitors video feeds to identify violent or aggressive behavior using deep learning models
- **Gender Recognition**: Identifies the gender of individuals in the frame to provide context-aware threat assessment
- **Gesture Recognition**: Detects distress signals and suspicious hand gestures that may indicate danger
- **Voice Analysis**: Analyzes audio patterns to detect distress calls or screaming
- **Automated Alert System**: Instantly notifies police stations and authorities when suspicious activity is detected
- **Multi-Modal Analysis**: Combines visual, audio, and behavioral cues for accurate threat detection
- **Continuous Monitoring**: 24/7 surveillance capability for enhanced security

## Technologies Used

### Core Technologies
- **Python 3.x**: Primary programming language
- **OpenCV (cv2)**: Computer vision and video processing
- **TensorFlow/Keras**: Deep learning framework for model training and inference
- **PyTorch**: Neural network implementation for violence detection

### Machine Learning & AI
- **MediaPipe**: Hand gesture recognition and pose detection
- **Deep Learning Models**:
  - Violence Detection Model (CNN-based)
  - Gender Detection Model (Keras)
  - Gesture Recognition Model
  - Voice Pattern Analysis Model

### Additional Libraries
- **NumPy**: Numerical computations and array operations
- **Joblib**: Model serialization and deserialization
- **SpeechRecognition**: Audio processing and voice analysis
- **scikit-learn**: Machine learning utilities and preprocessing

### Pre-trained Models
- `violence_detector.pth`: PyTorch model for violence detection
- `gender_detection.keras`: Gender classification model
- `gesture_recognition_model.h5`: Hand gesture recognition model
- `gesture_scaler.joblib`: Feature scaler for gesture data

## How It Works

1. **Video Capture**: The system captures video feed from connected cameras or webcams
2. **Frame Processing**: Each frame is processed through multiple AI models:
   - Gender detection identifies individuals in the frame
   - Violence detection analyzes actions and behaviors
   - Gesture recognition detects distress signals
3. **Audio Analysis**: Simultaneous audio monitoring detects screams or distress calls
4. **Threat Assessment**: The system combines all inputs to determine threat level
5. **Alert Generation**: When a threat is detected, automated alerts are sent to:
   - Nearby police stations
   - Emergency response teams
   - Registered authorities
6. **Continuous Monitoring**: The system continues monitoring and logging all activities

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- Webcam or IP camera access
- GPU recommended for real-time processing (optional but preferred)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Fayaz-y/Women-Safety-Empowerment-System.git
cd Women-Safety-Empowerment-System
git checkout Fayaz
```

2. **Install required dependencies**
```bash
pip install opencv-python
pip install tensorflow
pip install torch torchvision
pip install mediapipe
pip install numpy
pip install joblib
pip install SpeechRecognition
pip install scikit-learn
```

3. **Download pre-trained models**
Ensure all model files are in the root directory:
- `violence_detector.pth`
- `gender_detection.keras`
- `gesture_recognition_model.h5`
- `gesture_scaler.joblib`

4. **Configure alert settings**
Edit `keys.py` to add your alert system credentials and police station contact information

### Running the System

**Main Combined System:**
```bash
python combined_test.py
```

**Individual Modules:**
- Violence Detection: `python violence.py`
- Gender Detection: `python gender.py`
- Gesture Recognition: `python model.py`
- Voice Analysis: `python Voice_model.py`

## Alert Mechanism

The system implements a multi-channel alert mechanism:

1. **Threat Detection**: When suspicious activity is detected, the system calculates a threat score
2. **Alert Triggers**: High-confidence threats automatically trigger alerts
3. **Notification Channels**:
   - SMS/Text alerts to police stations
   - Email notifications to authorities
   - API calls to emergency response systems
   - Local alarm activation (if configured)
4. **Evidence Collection**: The system captures and stores:
   - Video footage of the incident
   - Screenshots of detected threats
   - Audio recordings (if applicable)
   - Timestamp and location data

**Alert Configuration**: Modify alert thresholds and contact information in `keys.py` to customize the notification system for your deployment.

## Contribution Guide

We welcome contributions from the community! Here's how you can help:

### Getting Started
1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/YourFeature`)
3. Make your changes and test thoroughly
4. Commit with clear messages (`git commit -m 'Add new feature: description'`)
5. Push to your branch (`git push origin feature/YourFeature`)
6. Open a Pull Request

### Areas for Contribution
- **Model Improvement**: Enhance accuracy of detection models
- **New Features**: Add support for additional threat detection methods
- **Optimization**: Improve processing speed and resource efficiency
- **Documentation**: Improve code documentation and user guides
- **Testing**: Write unit tests and integration tests
- **UI Development**: Create a user-friendly monitoring interface
- **Alert Systems**: Integrate with additional notification platforms

### Code Standards
- Follow PEP 8 style guidelines for Python code
- Add comments for complex logic
- Test your changes before submitting
- Update documentation as needed

## License

This project is open source and available for use in improving women's safety. Please ensure ethical use of this technology and comply with local privacy and surveillance laws.

---

**Note**: This system is designed to assist in safety monitoring and should be used as part of a comprehensive security strategy. Always ensure compliance with local laws regarding surveillance and data privacy.

**Contact**: For questions, issues, or collaboration opportunities, please open an issue on GitHub.

**Disclaimer**: This is a safety assistance tool and should not replace traditional security measures or emergency services. Always contact local authorities in case of immediate danger.
