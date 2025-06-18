import cv2
import matplotlib.pyplot as plt

def crop_fixed_borders(image, border_size=10):
    """Supprime une bordure fixe de 'border_size' pixels autour de l'image."""
    h, w = image.shape[:2]
    return image[border_size:h-border_size, border_size:w-border_size]  # Rognage

image_path = "C:/Users/h09184/Downloads/detection_classification_hologrammes/images/origins/passport/psp01_01_01/img_0039.jpg"

# Charger l'image
image = cv2.imread(image_path)
if image is None:
    print(" Erreur : l'image n'a pas pu être chargée. Vérifie le chemin.")
else:
    # Recadrage
    cropped_image = crop_fixed_borders(image, border_size=10)

    # Affichage avec Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Avant (avec bordures)")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Après (bordures enlevées)")
    axes[1].axis("off")

    plt.show()
    
print(image.shape)
print(cropped_image.shape)
# Charger le masque
mask = cv2.imread("passport_hologram_mask.png") 

# Redimensionner le masque à la taille souhaitée (595x412)
resized_mask = cv2.resize(mask, (595, 412))

# Enregistrer le masque redimensionné
cv2.imwrite("passport_hologram_mask_resized_412x595.png", resized_mask)  
