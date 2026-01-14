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
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation= 'relu'), #Learn combinations of features specific to motifs
    layers.Dropout(0.5), #Prevents overfitting
    layers.Dense(3, activation='softmax') #output - 5 batik motifs 
])

train_ds = keras.utils.image_dataset_from_directory (
    'data/train_augmented',
    validation_split=0.2,
    subset="both",
    image_size=(224, 224),
    batch_size=32,
    label_mode='int'
)

test_ds = keras.utils.image_dataset_from_directory(
    'data/test_processed',
    image_size=(224,224),
    batch_size=32,
    label_mode='int',
    shuffle=False
)

model.compile(optimizer=keras.optimizers.Adam(), 
              loss = keras.losses.BinaryCrossentropy(from_logits=True),
              metrics =[keras.metrics.BinaryAccuracy()])

model.fit(train_ds, epochs= 20)