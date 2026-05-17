# Core libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers, models
from tensorflow.keras import ops
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

#To stop model from overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience= 3,
    restore_best_weights=True
)

#Training Data
train_ds = keras.utils.image_dataset_from_directory (
    'data/train_augmented',
    image_size=(224, 224),
    batch_size=32,
    label_mode='int'
).map(apply_preprocessing)

#Validation Data
val_ds = keras.utils.image_dataset_from_directory (
    'data/validation',
    image_size=(224, 224),
    batch_size=16,
    label_mode='int'
).map(apply_preprocessing)

#Testing Data
raw_test_ds = keras.utils.image_dataset_from_directory(
    'data/test_processed',
    image_size=(224,224),
    batch_size=5,
    label_mode='int',
    shuffle=False
)

class_names = raw_test_ds.class_names
test_ds = raw_test_ds.map(apply_preprocessing)


model.compile(optimizer=keras.optimizers.Adam(), 
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), #frim_logits expects raw outputs (logits) or probabilities
              metrics =['accuracy'])

print("Model Summary:")
model.summary()

#Model Training
print("Training starts NOW...")
history = model.fit(
    train_ds,
    validation_data=val_ds, 
    epochs = 20,
    callbacks=[early_stop]
)

#Test Evaluation
print("Evaluate on test set NOW...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")


#Classification Report
y_true = np.concatenate([y for x, y in test_ds], axis=0) #True class labels
y_pred_probs = model.predict(test_ds) #Predicted
y_pred = np.argmax(y_pred_probs, axis = 1)

print("Classification Report: ")
print(classification_report(y_true, y_pred, target_names=['Batik Kawung', 'Batik Megamendung', 'Batik Parang']))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

cm_normalized = cm.astype('float') /cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6, 5))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#Save Model
model.save('batik_model.keras')
print()