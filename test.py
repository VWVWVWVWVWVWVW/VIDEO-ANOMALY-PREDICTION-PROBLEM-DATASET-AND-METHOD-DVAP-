from functions import compute_rbg_clips, compute_optical_flow_clips, i3d_module, fem_module
import tensorflow as tf
import numpy as np
import os
from keras.layers import Input, Conv1D
from keras.models import Model

c1 = tf.keras.models.load_model("Saved_Models/channel1_model.h5", compile=False)
c2 = tf.keras.models.load_model("Saved_Models/channel2_model.h5", compile=False)

def process_video(video_path):
    clips = compute_rbg_clips(video_path)
    optical_flow_clips = compute_optical_flow_clips(clips)
    return clips, optical_flow_clips

def extract_features(video_path):
    clips, optical_flow_clips = process_video(video_path)
    rgb_features, of_features, f_matrix = i3d_module(clips, optical_flow_clips)
    
    input_layer_w = Input(shape=(f_matrix.shape[1], 1))
    conv_layer_w = Conv1D(filters=1, kernel_size=1, padding='same', activation=None)(input_layer_w)
    model_w = Model(inputs=input_layer_w, outputs=conv_layer_w)
    
    input_layer_f = Input(shape=(f_matrix.shape[1], 1))
    conv_layer_f = Conv1D(filters=1, kernel_size=1, padding='same', activation=None)(input_layer_f)
    model_f = Model(inputs=input_layer_f, outputs=conv_layer_f)
    
    fem_matrix = fem_module(f_matrix, model_w, model_f)
    return fem_matrix

def predict_video(video_path):
    fem_matrix = extract_features(video_path)
    c1_output = c1.predict(fem_matrix)
    c2_output = c2.predict(fem_matrix)
    
    if np.mean(c1_output) > 0.5 and np.mean(c2_output) > 0.5:
        print(f"{video_path} - Anomaly detected with probability: {np.mean(c2_output):.4f}")
    else:
        print(f"{video_path} - Normal video")

test_video_folder = "Normal_Videos_for_Event_Recognition"  
test_video_paths = [os.path.join(test_video_folder, f) for f in os.listdir(test_video_folder) if f.endswith(".mp4")]

for test_video in test_video_paths:
    predict_video(test_video)