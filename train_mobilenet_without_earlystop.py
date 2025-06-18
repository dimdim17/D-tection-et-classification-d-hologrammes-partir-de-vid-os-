import os
import cv2
import pickle
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── 1. Dossiers ───────────────────────────────────────────────────────────
data_dir = 'images/Mosaics2_splited'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validation')
test_dir = os.path.join(data_dir, 'test')

# ─── 2. Prétraitement et Générateurs ───────────────────────────────────────
train_augmentor = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
    rotation_range=10,
    shear_range=20,
    horizontal_flip=True,
    vertical_flip=True
)

val_augmentor = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
)

train_generator = train_augmentor.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary'
)

val_generator = val_augmentor.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary'
)

# ─── 3. Chargement du modèle de base ───────────────────────────────────────
feature_extractor = tf.keras.applications.MobileNet(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False
)

freeze_ratio = 0.12
freeze_limit = int(len(feature_extractor.layers) * freeze_ratio)

for layer in feature_extractor.layers[:freeze_limit]:
    layer.trainable = False

# ─── 4. Construction du modèle complet ─────────────────────────────────────
classifier = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

classifier.build((None, 224, 224, 3))

# ─── 5. Compilation ────────────────────────────────────────────────────────
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

classifier.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

classifier.summary()

# ─── 6. Enregistrement et Entraînement ─────────────────────────────────────
output_dir = "mobilenet_custom_trained2"
os.makedirs(output_dir, exist_ok=True)

checkpoint_path = os.path.join(output_dir, 'best_model.h5')

model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

history = classifier.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    batch_size=64,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=[model_ckpt]
)

# Dernier modèle
classifier.save(os.path.join(output_dir, 'final_model.h5'))

# Sauvegarde historique
with open(os.path.join(output_dir, 'training_history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)

# ─── 7. Visualisation des courbes ──────────────────────────────────────────
def plot_learning_curves(metric, save_name):
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.lineplot(x=range(len(history.history[metric])), y=history.history[metric], marker='o', label='Train')
    sns.lineplot(x=range(len(history.history['val_' + metric])), y=history.history['val_' + metric], marker='o', label='Validation')
    ax.set_title(f'Courbe d\'apprentissage - {metric}')
    ax.set_ylabel(metric.capitalize())
    ax.set_xlabel('Épochs')
    ax.legend(loc='best')
    plt.savefig(os.path.join(output_dir, save_name))
    plt.show()

plot_learning_curves('loss', 'loss_curve.png')
plot_learning_curves('accuracy', 'accuracy_curve.png')
