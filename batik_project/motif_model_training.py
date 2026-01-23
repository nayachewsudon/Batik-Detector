# Core libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers, models
from tensorflow.keras import ops
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    layers.Dense(128, activation= 'relu'), 
    layers.Dropout(0.5), 
    layers.Dense(3) 
])

def apply_preprocessing(images, labels):
    """The augmentation script converts the images back to 255 pixel intensity value, range of [0, 255].
    This function applies preprocessing to change pixel intensity to [-1, 1]. MobileNetV2 accepts this range."""

    return preprocess_input(images), labels

train_ds = keras.utils.image_dataset_from_directory (
    'data/train_augmented',
    validation_split=0.2,
    subset="training",
    seed = 123,
    image_size=(224, 224),
    batch_size=32,
    label_mode='int'
).map(apply_preprocessing)

val_ds = keras.utils.image_dataset_from_directory (
    'data/train_augmented',
    validation_split=0.2,
    subset="validation",
    seed = 123,
    image_size=(224, 224),
    batch_size=32,
    label_mode='int'
).map(apply_preprocessing)

test_ds = keras.utils.image_dataset_from_directory(
    'data/test_processed',
    image_size=(224,224),
    batch_size=32,
    label_mode='int',
    shuffle=False
).map(apply_preprocessing)

model.compile(optimizer=keras.optimizers.Adam(), 
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), #frim_logits expects raw outputs (logits) or probabilities
              metrics =['accuracy'])

print("Model Summary:")
model.summary()

print("Training starts NOW...")
history = model.fit(
    train_ds,
    validation_data=val_ds, 
    epochs = 20
)

print("Evaluate on test set NOW...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

model.save('batik_model.keras')
print()