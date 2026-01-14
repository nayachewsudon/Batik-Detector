# Core libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers, models
from tensorflow.keras import ops
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

base_model = MobileNetV2(
    weights = 'imagenet',
    include_top = False,
    input_shape = (224, 224, 3)
)

#Freeze pretrained model; weights won't update during training
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(), #compresses spatial info into a single vector
    layers.Dense(128, activation= 'relu'), #Learn combinations of features specific to motifs
    layers.Dropout(0.5), #Drops 50% connections during training, prevent overfitting
    layers.Dense(5, activation='softmax') #output - 5 batik motifs 
])