import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json

# Charger une image pour vérifier sa taille
image_path = "passport_hologram_mask.png" 
image = cv2.imread(image_path)
if image is None:
    print("Erreur : l'image n'a pas pu être chargée.")
else:
    height, width = image.shape[:2]
    print(f"Taille de l'image : {width}x{height} pixels")

def find_optimal_patch_sizes(height, width):
    candidates = [1, 2, 4, 8, 10, 20, 25, 35, 40, 50, 100, 200]  
    best_height = max([p for p in candidates if height % p == 0], default=None)
    best_width = max([p for p in candidates if width % p == 0], default=None)
    return best_height, best_width

patch_height, patch_width = find_optimal_patch_sizes(height, width)

if patch_height and patch_width:
    print(f"Patch optimal : {patch_width}x{patch_height} pixels")
else:
    print("Aucune taille de patch idéale trouvée, il faudra gérer les bords.")

psp_width = 595
psp_height = 412

patch_height = 28
patch_width = 28
patch_size = (patch_height, patch_width)

num_rows = 8
num_cols = 8

number_of_patch = int((psp_height / patch_height) * (psp_width / patch_width))

def get_patch(patch_id, image_path):  
    image = cv2.imread(image_path)
    if image is None:
        return None
    height, width = image.shape[:2]

    cpt = 0
    for y in range(0, height, patch_height):
        for x in range(0, width, patch_width):
            if cpt == patch_id:
                patch = image[y:y + patch_height, x:x + patch_width]
                if patch.shape[:2] != (patch_height, patch_width):
                    patch = cv2.resize(patch, (patch_width, patch_height), interpolation=cv2.INTER_NEAREST)
                return patch
            cpt += 1
    return None

def get_mosaic(patch_id, frames_path):
    mosa = []
    for root, dirs, files in os.walk(frames_path):
        for file in sorted(files):
            if file.endswith(".jpg") or file.endswith(".png"):
                patch = get_patch(patch_id, os.path.join(root, file))
                if patch is not None:
                    mosa.append(patch)
    return mosa

def get_patches_mask(patches_mask):
    labels = []
    for patch in patches_mask:
        if np.all(patch == 0):
            labels.append('No-Holo')
        else:
            labels.append('Holo')
    return labels

def get_all_mosaics(current_path, labels_mask):
    mosaics2 = {'Holo': [], 'No-Holo': []}
    for k in range(number_of_patch):
        mosa = get_mosaic(k, current_path)
        patches_array = np.array(mosa)

        final_image = np.zeros((num_rows * patch_height, num_cols * patch_width, 3), dtype=np.uint8)
        
        for patch_index in range(min(len(mosa), num_rows * num_cols)):
            i = patch_index // num_cols
            j = patch_index % num_cols
            patch = patches_array[patch_index]
            final_image[i * patch_height: (i + 1) * patch_height, j * patch_width: (j + 1) * patch_width, :] = patch

        if len(mosa) > 0:
            mosaics2[labels_mask[k]].append(final_image)

    return mosaics2

def divide_image_into_patches(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    patch_height, patch_width = patch_size
    patches = []
    cpt = 0
    for y in range(0, height, patch_height):
        for x in range(0, width, patch_width):
            patch = image[y:y + patch_height, x:x + patch_width]
            patches.append(patch)
            cpt += 1
    return patches, cpt

def all_(mother_frames_path, save_path):
    cpt_glob = 0
    patches_mask, num_patches = divide_image_into_patches("passport_hologram_mask_resized_412x595.png")
    labels_mask = get_patches_mask(patches_mask)
    os.makedirs(os.path.join(save_path, "Holo"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "No-Holo"), exist_ok=True)
    
    for current_path, directories, files in os.walk(mother_frames_path):
        if files and 'psp' in current_path:
            print(cpt_glob, current_path)
            result = get_all_mosaics(current_path, labels_mask)
            for key, val in result.items():
                folder = os.path.join(save_path, key, f"psp_{str(cpt_glob).zfill(5)}")
                os.makedirs(folder, exist_ok=True)
                for cpt, img in enumerate(val):
                    cv2.imwrite(os.path.join(folder, f"mosa_{str(cpt).zfill(5)}.jpg"), img)
            cpt_glob += 1

if __name__ == '__main__':
    all_(
        "C:/Users/h09184/Downloads/detection_classification_hologrammes/images/output_seuillages", 
        "C:/Users/h09184/Downloads/detection_classification_hologrammes/images/Mosaics2_sans_repetition"
    )
