

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

