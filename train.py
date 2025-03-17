from functions import compute_rbg_clips, compute_optical_flow_clips, i3d_module, fem_module, channel
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import random
import os
from keras.layers import Conv1D, Input, LSTM, BatchNormalization, Dropout, Dense, ReLU
from keras.models import Model

epochs = 2
batch_size = 1
learning_rate = 0.001
weight_decay = 0.0005
alpha = 0.1

def process_video(video_path):
    clips = compute_rbg_clips(video_path)
    optical_flow_clips = compute_optical_flow_clips(clips)
    return clips, optical_flow_clips

def custom_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    continuity_loss = tf.reduce_sum(tf.abs(y_pred[:, 1:] - y_pred[:, :-1]))
    return mse_loss + continuity_loss

def build_model():
    input_shape = (2, 800) 
    c1 = channel(input_shape)
    c2 = channel(input_shape)
    return c1, c2

def train_model(c1, c2, dataset, labels, epochs=epochs, batch_size=batch_size):
    optimizer_c1 = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer_c2 = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    c1.compile(optimizer=optimizer_c1, loss=custom_loss, metrics=['accuracy'])
    c1.fit(dataset, labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    for layer in c1.layers:
        layer.trainable = False
    
    c2.compile(optimizer=optimizer_c2, loss=custom_loss, metrics=['accuracy'])
    c2.fit(dataset, labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)

def extract_features(video_path):
    clips, optical_flow_clips = process_video(video_path)
    rgb_features, of_features, f_matrix = i3d_module(clips, optical_flow_clips)
    
    input_layer_w = Input(shape=(f_matrix.shape[1], 1))
    conv_layer_w = Conv1D(filters=1, kernel_size=1, padding='same', activation=None)(input_layer_w)
    model_w = Model(inputs=input_layer_w, outputs=conv_layer_w)
    
    input_layer_f = Input(shape=(f_matrix.shape[1], 1))
    conv_layer_f = Conv1D(filters=1, kernel_size=1, padding='same', activation=None)(input_layer_f)
    model_f = Model(inputs=input_layer_f, outputs=conv_layer_f)
    
    fem_matrix = fem_module(f_matrix, model_w, model_f, alpha=alpha)
    return fem_matrix

y_true = np.zeros(2) 
video_folder = "Normal_Videos_for_Event_Recognition" 
video_paths = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(".mp4")]
video_paths = video_paths[:5]  

dataset = np.array([extract_features(vp) for vp in video_paths])

dataset = np.squeeze(dataset, axis=1)  
print("Dataset shape after squeezing:", dataset.shape)

c1, c2 = build_model()
train_model(c1, c2, dataset[:2], y_true)

c1.save("Saved_Models/channel1_model.h5")
c2.save("Saved_Models/channel2_model.h5")