import os
import shutil
import random

# Définir les répertoires source et de sortie
main_folder = "C:/Users/h09184/Downloads/detection_classification_hologrammes/images/Mosaics2_sans_repetition"
output_folder = "C:/Users/h09184/Downloads/detection_classification_hologrammes/images/Mosaics2_sans_repetition_splited"
train_folder = os.path.join(output_folder, "train")
val_folder = os.path.join(output_folder, "validation")
test_folder = os.path.join(output_folder, "test")

# Créer les répertoires de sortie
for folder in [train_folder, val_folder, test_folder]:
    os.makedirs(os.path.join(folder, "Holo"), exist_ok=True)
    os.makedirs(os.path.join(folder, "No-Holo"), exist_ok=True)

# Parcourir les dossiers avec le même nom dans Holo et No-Holo
holo_folders = set(os.listdir(os.path.join(main_folder, "Holo")))
no_holo_folders = set(os.listdir(os.path.join(main_folder, "No-Holo")))
common_folders = holo_folders.intersection(no_holo_folders)

# Convertir common_folders en liste pour pouvoir l'utiliser avec random.shuffle()
common_folders = list(common_folders)

# Séparer les dossiers en ensembles de train, validation et test
random.shuffle(common_folders)
num_common_folders = len(common_folders)

num_train = int(num_common_folders * 0.8)
num_val = int(num_common_folders * 0.1)
num_test = num_common_folders - num_train - num_val

train_folders = common_folders[:num_train]
val_folders = common_folders[num_train:num_train + num_val]
test_folders = common_folders[num_train + num_val:]

# Distribuer les dossiers avec un nom commun en test, validation et train
for folder_name in train_folders:
    shutil.copytree(os.path.join(main_folder, "Holo", folder_name), os.path.join(train_folder, "Holo", folder_name))
    shutil.copytree(os.path.join(main_folder, "No-Holo", folder_name), os.path.join(train_folder, "No-Holo", folder_name))

for folder_name in val_folders:
    shutil.copytree(os.path.join(main_folder, "Holo", folder_name), os.path.join(val_folder, "Holo", folder_name))
    shutil.copytree(os.path.join(main_folder, "No-Holo", folder_name), os.path.join(val_folder, "No-Holo", folder_name))

for folder_name in test_folders:
    shutil.copytree(os.path.join(main_folder, "Holo", folder_name), os.path.join(test_folder, "Holo", folder_name))
    shutil.copytree(os.path.join(main_folder, "No-Holo", folder_name), os.path.join(test_folder, "No-Holo", folder_name))

# Distribuer les dossiers restants en train, validation et test avec le ratio 80-10-10
remaining_holo_folders = holo_folders.difference(common_folders)
remaining_no_holo_folders = no_holo_folders.difference(common_folders)

remaining_folders = list(remaining_holo_folders.union(remaining_no_holo_folders))
random.shuffle(remaining_folders)

num_train_remaining = int(len(remaining_folders) * 0.8)
num_val_remaining = int(len(remaining_folders) * 0.1)
num_test_remaining = len(remaining_folders) - num_train_remaining - num_val_remaining

train_remaining_folders = remaining_folders[:num_train_remaining]
val_remaining_folders = remaining_folders[num_train_remaining:num_train_remaining + num_val_remaining]
test_remaining_folders = remaining_folders[num_train_remaining + num_val_remaining:]

for folder_name in train_remaining_folders:
    shutil.copytree(os.path.join(main_folder, "No-Holo", folder_name), os.path.join(train_folder, "No-Holo", folder_name))

for folder_name in val_remaining_folders:
    shutil.copytree(os.path.join(main_folder, "No-Holo", folder_name), os.path.join(val_folder, "No-Holo", folder_name))

for folder_name in test_remaining_folders:
    shutil.copytree(os.path.join(main_folder, "No-Holo", folder_name), os.path.join(test_folder, "No-Holo", folder_name))

