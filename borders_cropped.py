import cv2
import os
import numpy as np

def crop_fixed_borders(image, border_size=10):
    """Supprime une bordure fixe de 'border_size' pixels autour de l'image."""
    h, w = image.shape[:2]
    return image[border_size:h-border_size, border_size:w-border_size]  # Rognage

def process_dataset(input_root, output_root, border_size=10):
    """Parcourt le dataset et applique le recadrage fixe sur toutes les images."""
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith((".jpg", ".png")):  # Vérifier les formats d'image
                input_path = os.path.join(root, file)
                output_dir = root.replace(input_root, output_root, 1)  # Recréer la structure du dossier
                os.makedirs(output_dir, exist_ok=True)
                
                image = cv2.imread(input_path)
                if image is not None:
                    cropped_image = crop_fixed_borders(image, border_size)
                    output_path = os.path.join(output_dir, file)
                    cv2.imwrite(output_path, cropped_image)

    print(" Recadrage terminé : toutes les images sont sauvegardées sans les bordures.")

if __name__ == '__main__':
    input_dataset = "C:/Users/h09184/Downloads/detection_classification_hologrammes/images/fraud/copy_without_holo"
    output_dataset = "C:/Users/h09184/Downloads/detection_classification_hologrammes/images/fraud/copy_without_holo/passport_cropped"

    process_dataset(input_dataset, output_dataset, border_size=10)
