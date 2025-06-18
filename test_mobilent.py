import os
import pickle
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── 1. Chargement des données de test ─────────────────────────────────────
test_dir = os.path.join('images/Mosaics2_splited', 'test')

test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    class_mode='binary',
    batch_size=1,
    shuffle=False
)

# ─── 2. Chargement du modèle entraîné ──────────────────────────────────────
model_path = os.path.join('mobilenet_custom_trained2', 'best_model.h5')
model = tf.keras.models.load_model(model_path)
model.summary()

# ─── 3. Évaluation du modèle ───────────────────────────────────────────────
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Loss     : {test_loss:.4f}")

# ─── 4. Prédictions ────────────────────────────────────────────────────────
test_generator.reset()
probas = model.predict(test_generator).squeeze()
preds = (probas >= 0.5).astype(int)
true_labels = test_generator.classes

# ─── 5. Matrice de confusion ───────────────────────────────────────────────
cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Vérités terrain")
plt.savefig(os.path.join('mobilenet_custom_trained2', 'confusion_matrix.png'))
plt.show()

# ─── 6. Rapport de classification ─────────────────────────────────────────
print("\nRapport de classification :\n")
print(classification_report(true_labels, preds, digits=4))

# ─── 7. Sensibilité / Spécificité ──────────────────────────────────────────
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

print(f"Sensibilité : {sensitivity:.4f}")
print(f"Spécificité : {specificity:.4f}")

# ─── 8. Courbe ROC ─────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(true_labels, probas)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC")
plt.legend(loc='lower right')
plt.savefig(os.path.join('mobilenet_custom_trained2', 'roc_curve.png'))
plt.show()
