import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
import numpy as np
import cv2
import numpy as np
import random
from keras.layers import Conv1D, Input
from keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Conv1D, BatchNormalization, Dropout, Dense, ReLU
from tensorflow.keras.models import Model
import numpy as np


i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

def compute_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def compute_rbg_clips(video_path):
    taille_clip = 3
    num_clips = 2
    target_size = (224, 224)
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    nbr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # fpc de 3sec
    fpc = taille_clip * fps

    start_frame = random.randint(0, nbr_frames - fpc)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(fpc):
        end_video, frame = cap.read()
        if not end_video:
            break
        frame = cv2.resize(frame, target_size) / 255.0
        frames.append(frame)
    cap.release()

    clip = np.array(frames, dtype=np.float32)

    frames_per_clip = len(clip) // num_clips
    clips = [clip[i * frames_per_clip:(i+1) * frames_per_clip] for i in range(num_clips)]#sliding winodws

    return np.array(clips ,dtype=np.float32)

def compute_optical_flow_clips(clips):
    optical_flow_clips = []
    for clip in clips:
        optical_flows = []
        for i in range(len(clip) - 1):
            flow = compute_optical_flow((clip[i]).astype(np.uint8), (clip[i + 1]).astype(np.uint8))
            #nzido 3rd dim avec zeros pour que le model i3d peuvent les traiter
            flow = np.concatenate([flow, np.zeros((flow.shape[0], flow.shape[1], 1))], axis=-1)
            optical_flows.append(flow)
        #nzido des 0 de pad ftali bech yg3dolna nefss les tailes, (bech tkhdmlna 32,2048)
        optical_flows.append(np.zeros_like(optical_flows[0]))
        optical_flow_clips.append(np.array(optical_flows, dtype=np.float32))

    return np.array(optical_flow_clips, dtype=np.float32)




def i3d_module(clips, optical_flow_clips):

    rgb_inputs = tf.convert_to_tensor(clips, dtype=tf.float32)
    of_inputs = tf.convert_to_tensor(optical_flow_clips, dtype=tf.float32)

    rgb_outputs = i3d(rgb_inputs)
    of_outputs = i3d(of_inputs)

    rgb_features = rgb_outputs['default'].numpy()
    of_features = of_outputs['default'].numpy()

    f_matrix = np.concatenate((rgb_features, of_features), axis=-1)

    return rgb_features, of_features, f_matrix



def fem_module(f_matrix, model_w, model_f, alpha=0.4):
    w_matrix = np.sqrt(np.sum(f_matrix ** 2, axis=1, keepdims=True))

    w_conv = model_w.predict(w_matrix)
    f_conv = model_f.predict(f_matrix)

    f_conv_reshaped = np.transpose(f_conv, (2, 0, 1))
    w_conv_expanded = np.expand_dims(w_conv, axis=-1)
    w_conv_repeated = np.repeat(w_conv_expanded, f_matrix.shape[1], axis=2)
    w_conv_squeezed = np.squeeze(w_conv_repeated, axis=-1)
    w_conv_reshaped = np.transpose(w_conv_squeezed, (1, 0, 2))

    fem = f_conv_reshaped + alpha * w_conv_reshaped

    return fem / np.linalg.norm(fem, axis=-1, keepdims=True)

    

def channel(input_shape, lstm_units=128, conv_filters=64, kernel_size=3, dropout_rate=0.3, dense_units=128, output_units=1, activation='sigmoid'):
    
    input_layer = Input(shape=input_shape)
    lstm_layer = LSTM(lstm_units, return_sequences=True)(input_layer)

    conv_layer = Conv1D(filters=conv_filters, kernel_size=kernel_size, padding='same', activation=None)(lstm_layer)
    norm_layer = BatchNormalization()(conv_layer)

    drop_layer = Dropout(dropout_rate)(norm_layer)
    dense1 = Dense(dense_units)(drop_layer)
    relu1 = ReLU()(dense1)

    drop_layer = Dropout(dropout_rate)(relu1)
    dense2 = Dense(dense_units)(drop_layer)
    relu3 = ReLU()(dense2)

    dense3 = Dense(dense_units)(relu3)
    output_layer = Dense(output_units, activation=activation)(dense3)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def dom_module(fem_matrix, lstm_units=128, conv_filters=64, kernel_size=3, dropout_rate=0.3, dense_units=128, output_units=1, activation='sigmoid'):
    input_shape = fem_matrix.shape[1:]

    c1 = channel(input_shape, lstm_units, conv_filters, kernel_size, dropout_rate, dense_units, output_units, activation)
    c2 = channel(input_shape, lstm_units, conv_filters, kernel_size, dropout_rate, dense_units, output_units, activation)
    
    c1_output = c1.predict(fem_matrix)
    c2_output = c2.predict(fem_matrix)
    
    if np.mean(c1_output) > 0.5 and np.mean(c2_output) > 0.5:
        print("This video is an Anomaly, and the percentage of it happening in the next second is : ",c2_output)
    else:
        print("Normal video")

