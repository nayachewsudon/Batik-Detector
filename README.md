# Batik Detector Web App

### About the Project
Batik is an Indonesian craft which uses wax-resistant dyes to fabric in order to create beautifully intricate patterns.  With about 5.849 batik motifs in Indonesia as of today, Indonesian batik was added to the UNESCO's Intangible Cultural Heritage of Humanity List in 2009. 

This web app detects batik patterns from images uploaded on the website and classifies the batik pattern. Currently, the web app is trained on a dataset of three popular batik motifs: Batik Megamendung, Batik Kawung, and Batik Parang, all carrying significant meaning and culture behind its patterns. 

![Alt text](images/batik-parang.jpg)

*Batik Parang*

![Alt text](images/batik-kawung.jpg)

*Batik Kawung*

![Alt text](images/batik-megamendung.jpg)

*Batik Megamendung*

### Model Architecture
The dataset was trained on MobileNetV2, a light CNN model provided by Tensorflow.
- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Custom Layers:** 
  - GlobalAveragePooling2D
  - Dense(128, activation='relu')
  - Dropout(0.5)
  - Dense(3) output layer
- **Input Size:** 224×224×3
- **Loss Function:** Sparse Categorical Crossentropy (from logits)
- **Optimizer:** Adam (learning rate: 0.001)

### Under Construction: 

- Grad-CAM implementation to identify regions of the image used to classify as a certain batik pattern.
- Front-end of the web application

## Acknowledgments

- Dataset: [dionisiusdh/indonesian-batik-motifs](https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs) on Kaggle
- Model: MobileNetV2 from TensorFlow/Keras
- Inspiration: Preserving Indonesian cultural heritage through AI
