import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import tensorflow as tf
import pickle
import numpy as np
import cv2

# Dossiers pour les ensembles de données
train_folder = 'images/Mosaics2_splited/train'
val_folder = 'images/Mosaics2_splited/validation'
test_folder = 'images/Mosaics2_splited/test'

# Data augmentation
train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input, 
                                   rotation_range=10,
                                   shear_range=20,
                                   horizontal_flip=True,
                                   vertical_flip=True)

val_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)

# Chargement des ensembles de données
training_set = train_datagen.flow_from_directory(train_folder,
                                                 target_size=(224, 224),
                                                 batch_size=64,
                                                 class_mode='binary')

validation_set = val_datagen.flow_from_directory(val_folder,
                                                 target_size=(224, 224),
                                                 batch_size=64,
                                                 class_mode='binary')

# Définition du modèle ViT Small (non pré-entraîné)
class VisionTransformer(tf.keras.Model):
    def __init__(self, num_classes=1):
        super(VisionTransformer, self).__init__()

        # Patch embedding
        self.patch_size = 16
        self.num_patches = (224 // self.patch_size) ** 2
        self.embed_dim = 256
        self.num_heads = 4
        self.num_layers = 4
        self.mlp_dim = 512

        # Embedder : découpe l'image en patches et les transforme
        self.embedding = tf.keras.layers.Dense(self.embed_dim)

        # Transformer blocks
        self.transformer_blocks = [
            tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
            for _ in range(self.num_layers)
        ]

        # Global Average Pooling
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()

        # MLP pour la classification
        self.dense1 = tf.keras.layers.Dense(self.mlp_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(num_classes, activation="sigmoid")

    def call(self, inputs):
        # Crée les patches
        patches = self.create_patches(inputs)
        x = self.embedding(patches)

        # Passe les données à travers chaque couche de transformer
        for transformer in self.transformer_blocks:
            x = transformer(x, x)
        
        # Moyenne globale pour réduire la dimension
        x = self.global_avg_pool(x)

        # Passer à travers MLP pour classification
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def create_patches(self, images):
        """Découpe les images en patches"""
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patches = tf.reshape(patches, (-1, self.num_patches, self.embed_dim))
        return patches

# Instancier et compiler le modèle
model = VisionTransformer(num_classes=1)

# Compiler le modèle
learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.build(input_shape=(None, 224, 224, 3))

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['acc']
)

model.summary()


# Définir le chemin de sauvegarde du meilleur modèle
filepath = "ViT_Model_no_weights/best_model.h5"

# Définir le callback ModelCheckpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Entraînement du modèle
history = model.fit(training_set, validation_data=validation_set, epochs=50, batch_size=64,
                    steps_per_epoch=len(training_set),
                    validation_steps=len(validation_set),
                    callbacks=[checkpoint])

# Sauvegarde du modèle final
model.save('ViT_Model_no_weights/last_epoch_model.h5')

# Sauvegarde de l'historique d'entraînement
with open('ViT_Model_no_weights/history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

# Courbes d'apprentissage : Perte (Loss)
fig, ax = plt.subplots(figsize=(20, 8))
sns.lineplot(x=history.epoch, y=history.history['loss'], marker="o")
sns.lineplot(x=history.epoch, y=history.history['val_loss'], marker="o")
ax.set_title('Learning Curve (Loss)')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
ax.legend(['train', 'val'], loc='best')
plt.savefig('ViT_Model_no_weights/Learning Curve (Loss).png')
plt.show()

# Courbes d'apprentissage : Précision (Accuracy)
fig, ax = plt.subplots(figsize=(20, 8))
sns.lineplot(x=history.epoch, y=history.history['acc'], marker="o")
sns.lineplot(x=history.epoch, y=history.history['val_acc'], marker="o")
ax.set_title('Learning Curve (Accuracy)')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epoch')
ax.legend(['train', 'val'], loc='best')
plt.savefig('ViT_Model_no_weights/Learning Curve (Accuracy).png')
plt.show()
