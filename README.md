# Chess Robot

An autonomous chess-playing robot that combines computer vision and robotic control to play physical chess games.

## Overview

This project implements a robotic arm that identifies chess pieces and their positions using computer vision, then physically moves the pieces using precise servo control and inverse kinematics. Significant progress has been made on the computer vision system, which now robustly detects the chessboard through a multi-step image processing pipeline.

## Hardware

- 20-inch articulated robotic arm (Modeled in Fusion360 and 3D printed)
- Raspberry Pi 4B
- Raspberry Pi AI Camera
- PCA9685 PWM driver
- High-torque 270° metal gear servos (40KG-60KG)

## Software

- **Computer Vision:** OpenCV-based pipeline for board detection
- **Machine Learning:** PyTorch neural network for piece classification
- **Robotics:** Inverse kinematics for precise arm movement
- **CAD:** 3D models created in Autodesk Fusion 360

## CAD Model

The robotic arm was designed and assembled in Autodesk Fusion 360. Below are some views of the CAD model:
### Home View
<img src="https://github.com/user-attachments/assets/7fde5c12-ea80-4ea5-99b3-357459eb46b3" alt="Fusion 360 Home View" width="300" height="auto">

### Front View
<img src="https://github.com/user-attachments/assets/513f0785-d334-4f0b-9211-2e1e1d9917d5" alt="Fusion 360 Front View" width="300" height="auto">

### Side View
<img src="https://github.com/user-attachments/assets/5dd077a4-1d71-4719-bd61-001bc527a5f2" alt="Fusion 360 Side View" width="300" height="auto">


## Computer Vision Process

The vision system processes the input image in several stages:

1. **Input Image**  
   The raw input captured by the AI camera.  
   <img src="images/3.JPG" alt="Input Image" width="300" height="auto">

2. **Preprocessing**  
   The image is resized, denoised `fastNlMeansDenoisingColored`, and smoothed with a bilateral filter to preserve edges.  
   <img src="https://github.com/user-attachments/assets/24b98eb7-3a1d-404f-a388-d4700c96671d" alt="Preprocessed Image" width="300" height="auto">

3. **Thresholding**  
   The image is converted to the HSV color space, and a mask is generated using predefined brown thresholds to isolate the chessboard region.  
   <img src="https://github.com/user-attachments/assets/5b3c4cd6-be36-4d1d-a3ee-7369d3b62b95" alt="Threshold Mask" width="300" height="auto">

4. **Morphological Operations**  
   Dilation, erosion, and closing `MORPH_CLOSE` are applied to refine the mask and reduce noise.  
   <img src="https://github.com/user-attachments/assets/b21a067a-233d-4ae9-bbd7-1963be2e80d4" alt="Morphological Processing" width="300" height="auto">

5. **Contour Detection & Polynomial Approximation**  
   The largest contour is detected and approximated to a quadrilateral using the `cv2.approxPolyDP` method, outlining the chessboard.  
   <img src="https://github.com/user-attachments/assets/e8a41df2-6813-4347-9032-f9421ea264e7" alt="Board Contour Approximation" width="300" height="auto">

6. **Perspective Transformation**  
   The detected board is warped into a square play area.  
   <img src="https://github.com/user-attachments/assets/4776e551-e417-49d0-b417-d2155ef867d4" alt="Warped & Cropped Board" width="300" height="auto">
   
7. **Square Separation**  
   Once the board is found and warped, it is cropped by a border margin and divided into an 8x8 grid that will be fed into the CNN.  
   <img src="https://github.com/user-attachments/assets/c2f5f91c-1d2b-4dab-b679-8f6e809fcbb5" alt="Warped Grid" width="300" height="auto">

## Progress & Future Work

- **Completed:**  
  - Robust board detection via multi-stage preprocessing and contour approximation.
  - Successful segmentation of the board into individual squares.
  - Trained Convolutional Neural Network (CNN) for square classification.
  - Stockfish API integration.
    
- **Planned:**  
  - Full integration of the inverse kinematics.
  - Further improvements in piece classification.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/zthoffman21/chessRobot.git
