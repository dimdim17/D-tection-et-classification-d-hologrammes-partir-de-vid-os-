import cv2
import os
import numpy as np
import csv

def detect_overexposed_region(image_path, output_folder, threshold=255):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Masque des pixels surexposés
    mask = (gray >= threshold).astype(np.uint8) * 255
    area = cv2.countNonZero(mask)

    # Préparation des chemins
    filename = os.path.basename(image_path)
    name, _ = os.path.splitext(filename)
    mask_path = os.path.join(output_folder, f"{name}_mask.png")

    # Enregistrement du masque
    cv2.imwrite(mask_path, mask)

    return filename, area
# === Dossiers ===
input_folder = "C:/Users/h09184/Downloads/detection_classification_hologrammes/images/origins/passport/psp07_03_03"
output_folder = "C:/Users/h09184/Downloads/detection_classification_hologrammes/images/output"
os.makedirs(output_folder, exist_ok=True)

# === Traitement ===
csv_path = os.path.join(output_folder, "overexposed_areas.csv")
with open(csv_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Filename", "Overexposed Area (pixels)"])

    for fname in sorted(os.listdir(input_folder)):
        if fname.endswith((".jpg", ".png")):
            full_path = os.path.join(input_folder, fname)
            filename, area = detect_overexposed_region(full_path, output_folder)
            writer.writerow([filename, area])
            print(f"{filename} → zone surexposée : {area} px")

print("Traitement terminé. Masques et CSV enregistrés.")
