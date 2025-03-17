

# Input Module :
## Challenges with Anomalous Videos
The input module was designed for normal videos because:

**Dataset Limitations**: The anomalous dataset was unavailable, and manually annotating the exact start time of anomalies in each video is time-consuming.

**Partial Dataset**: A subset of the UCF Crime dataset was used, but it lacked FPS information, requiring manual calculation of frames per clip.

**Clip Limitations**: The I3D model requires 32-frame clips, but the dataset provided insufficient frames, so clips were split into smaller segments.

**Optical Flow Requirement**: The original implementation did not mention optical flow, but it was later found to be essential for the model.

## How does the Input Module Works :
The input module processes videos to extract RGB frames and optical flow, which are used as inputs for the I3D model. Here's a brief explanation of the key functions:

**compute_optical_flow(prev_frame, next_frame):**

Computes the optical flow between two consecutive frames using the Farneback method.

Converts frames to grayscale and calculates motion vectors.

**compute_rbg_clips(video_path):**

Extracts RGB clips from a video.

Randomly selects a start frame and extracts a sequence of frames, resizing them to 224x224 and normalizing pixel values.

Splits the frames into smaller clips for the I3D model.

**compute_optical_flow_clips(clips):**

Computes optical flow for each pair of consecutive frames in the RGB clips.

Adds padding to match the number of frames and concatenates the flow with zeros to make it compatible with the I3D model.

## How does the I3D Module Works :

The i3d_module extracts spatio-temporal features from RGB video clips and optical flow clips using the I3D model. It processes both inputs separately, retrieves their features, and then concatenates them to form a fused representation (f_matrix), enhancing action recognition performance.

### Feature Enhancement Module (FEM)

After extracting the feature matrix \( F \) from the **I3D model**, the **FEM** refines these features using **1D Convolution** and **normalization**. The process follows these steps:

![image](https://github.com/user-attachments/assets/b217c7fc-9c29-41dd-ad45-7e535bc2c14a)

![image](https://github.com/user-attachments/assets/9095c524-59bb-42ff-b920-7addf540d5b7)


![image](https://github.com/user-attachments/assets/4da5a8a2-e1a1-45e3-81a2-8807d037594e)


# DOM Module Explanation :
The DOM (Dual-Output Model) module processes the FEM (Feature Enhancement Module) matrix using two parallel deep learning models (c1 and c2), each predicting the likelihood of an anomaly in the video.

### Feature Extraction:

The input to the DOM module is the FEM matrix obtained after passing video frames through the I3D model and FEM module.
The shape of this matrix is (B, N, T, C), containing temporal and spatial information.
Parallel Channel Processing:

### Two identical models (c1 and c2) process the FEM matrix.
Each channel consists of:
LSTM (for capturing temporal dependencies).
Conv1D + BatchNorm (for feature extraction).
Dense Layers with ReLU activation.
Dropout layers to prevent overfitting.
Final Dense Layer with Sigmoid activation, providing an anomaly probability.
Decision Making:

Both models (c1 and c2) generate predictions.
If both outputs exceed 0.5, the system detects an anomaly and predicts the anomaly probability for the next second.
Otherwise, the video is classified as normal.




This module enhances the representation of temporal dynamics and motion patterns, improving video understanding.

## Loss Function Explanation :
The loss function consists of two terms:

![image](https://github.com/user-attachments/assets/3ad718b6-9107-44fa-9158-4e30622d7202)

First Term (MSE - Mean Squared Error):
Measures how close the predicted values are to the actual labels.
Helps the model learn accurate predictions.
Second Term (Temporal Smoothness Regularization):
Enforces smoothness between consecutive frames.
Reduces abrupt changes in predictions by penalizing large variations.
Ensures consistent anomaly detection over time.

This combination improves both accuracy and temporal consistency in anomaly detection.


