import cv2
import os
import numpy as np

def get_neighboring_frames(filenames, index):
    prev_img = cv2.imread(filenames[index - 1]) if index > 0 else None
    next_img = cv2.imread(filenames[index + 1]) if index < len(filenames) - 1 else None
    return prev_img, next_img

def correct_overexposed_regions(current_img, mask, prev_img, next_img):
    corrected_img = current_img.copy()

    if prev_img is not None and next_img is not None:
        avg_img = cv2.addWeighted(prev_img, 0.5, next_img, 0.5, 0)
    elif prev_img is not None:
        avg_img = prev_img
    elif next_img is not None:
        avg_img = next_img
    else:
        return current_img  # Rien à corriger sans voisins

    for c in range(3):  # Pour chaque canal (BGR)
        corrected_img[:, :, c][mask == 255] = avg_img[:, :, c][mask == 255]

    return corrected_img

def process_folder(input_folder, output_folder, kernel_size=5):
    os.makedirs(output_folder, exist_ok=True)
    filenames = sorted([
        os.path.join(input_folder, f) for f in os.listdir(input_folder)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    for i, file_path in enumerate(filenames):
        print(f"Traitement : {file_path}")
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Détection des zones surexposées
        mask = (gray >= 255).astype(np.uint8) * 255

        # Dilatation (ou érosion selon ce que tu préfères)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        processed_mask = cv2.dilate(mask, kernel, iterations=1)

        # Correction par interpolation entre les frames voisines
        prev_img, next_img = get_neighboring_frames(filenames, i)
        corrected_img = correct_overexposed_regions(image, processed_mask, prev_img, next_img)

        # Enregistrement de l’image corrigée
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, corrected_img)

    print("Toutes les images ont été corrigées et enregistrées.")

# === Utilisation ===
input_folder = "C:/Users/h09184/Downloads/detection_classification_hologrammes/images/origins/passport/psp07_03_03"
output_folder = "C:/Users/h09184/Downloads/detection_classification_hologrammes/images/output1"
process_folder(input_folder, output_folder)
