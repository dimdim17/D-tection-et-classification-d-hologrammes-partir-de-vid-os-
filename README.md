**Détection et classification d’hologrammes (à partir de vidéos)**

**Présentation du projet - Approche par patchs & mosaiques**

Ce projet vise à détecter les hologrammes présents sur des passeports du dataset **[MIDV-HOLO](https://github.com/SmartEngines/midv-holo)**, puis à les classer en deux catégories : "Holo"  et "No-Holo". Pour cela, nous combinons des techniques de deep learning et de vision par ordinateur, en testant deux architectures de modèles : MobileNet et un Small Vision Transformer (ViT). 
- Dans un premier temps, un prétraitement est appliqué aux images afin d'extraire uniquement les zones contenant les passeports, ce qui permet de réduire les variations de fond à l’aide d’une transformation par homographie. Ensuite, un seuillage est utilisé pour corriger la surexposition due aux flashs. Les images ainsi traitées sont découpées en patchs, puis assemblées sous forme de mosaïques, qui servent de données d’entrée pour l’entraînement des modèles. 
- Une fois les modèles entraînés, nous les utilisons pour prédire, à partir de vidéos, la classe de chaque patch : les zones détectées comme contenant un hologramme sont colorées en vert, les autres sont colorées en rouge. Ce processus permet de localiser précisément les hologrammes dans les vidéos. 
- Enfin, pour chaque vidéo, nous calculons un ratio de patchs verts (c’est-à-dire détectés comme "Holo"). Ce ratio permet de définir automatiquement un seuil de classification entre les deux classes. Ce seuil est établi comme la moyenne entre : le minimum des ratios observés dans les vidéos Holo, et le maximum des ratios observés dans les vidéos No-Holo. Cette approche permet une classification fine, visuelle et robuste des passeports. 

**Pipeline de traitement et de classification**
<p align="center">
  <img src="https://github.com/user-attachments/assets/e65153c5-cf6c-44fd-9368-182ccd13ba7d" alt="Mosaïque" width="600"/>
</p>

- Homographie : À partir des coordonnées détectées des passeports dans les vidéos, nous avons appliqué une transformation par homographie pour extraire et redresser les passeports. Cette étape permet de standardiser les images en supprimant les variations de fond, d’échelle et de perspective, assurant ainsi une meilleure cohérence pour les traitements ultérieurs. 
- Traitement des frames par seuillage hystérésis : Nous avons adapté un seuillage par hystérésis pour détecter les zones surexposées. Un premier seuillage avec un seuil élevé (S1 > 240) permet d’isoler la grande composante connexe (CC1) correspondant à la zone très blanche. Un second seuillage, plus tolérant (S2 > 235), est ensuite appliqué à l’image d’origine pour extraire davantage de composantes connexes. Seules les composantes qui touchent CC1 sont conservées, ce qui permet d’étendre la détection à des zones adjacentes (par exemple, le halo lumineux). 
Une fois traitées, ces images sont crop pour réduire les bords noirs qui peuvent apparaitre (-10px) 
- Découpage en patchs : Les images redressées des passeports sont découpées en petits patchs de 28×28 pixels. Ces patchs sont ensuite réorganisés sous forme de mosaïques de taille fixe (8 × 28 = 224 pixels) afin de correspondre au format d’entrée attendu par le modèle. Ce découpage permet une analyse fine et localisée des différentes zones du document. Lorsque le nombre de patchs disponibles est insuffisant pour remplir complètement une mosaïque (en raison d’un nombre de frames variable selon les vidéos), les patchs existants sont répétés. Cette stratégie permet d’éviter l’apparition de zones noires dans les mosaïques, garantissant ainsi une entrée cohérente pour le modèle. 
- Classification des mosaïques : Chaque patch issu des mosaïques est ensuite classifié individuellement par le modèle, selon deux classes possibles : "Holo" (présence d’un hologramme) ou "No-Holo" (absence), en utilisant le masque binaire. 
- Entraînement des modèles de classification de mosaiques :

- **Modèle pré-entraîné Mobilenet : f1score : 91,86%**

![confusion_matrix](https://github.com/user-attachments/assets/4847c6cd-b5be-4dd4-80cc-fcb57964368a) ![loss_curve](https://github.com/user-attachments/assets/02535e95-af74-44cf-af46-b21d4f285268)

- **ViT small patch16 224: f1score : 95,92%**

![confusion_matrix](https://github.com/user-attachments/assets/c382ae70-03c4-4884-9a4d-9495dbbe429b)



- Les prédictions sont utilisées pour colorer les patchs sur les images des vidéos d’origine : les patchs "Holo" sont colorés en vert, les patchs "No-Holo" en rouge. Cette visualisation permet de localiser clairement les zones détectées comme holographiques. 
- Calcul du ratio de patchs verts : Pour chaque passeport, nous calculons le ratio de patchs verts parmi tous les patchs colorés. Ce ratio reflète la proportion d’hologramme détecté : un ratio élevé suggère une forte présence d’hologramme, un ratio faible indique une absence probable. 
-Détermination du seuil de classification :  déterminer un seuil de décision optimal en se basant sur le min/max des ratios observés entre les classes "Holo" et "No-Holo", et une régression logistique pour modéliser la séparation entre les deux distributions. 
-Classification finale : Les passeports sont ensuite classés automatiquement en "Holo" ou "No-Holo" selon que leur ratio dépasse ou non le seuil déterminé. 
-Résultats de Classification : holo (origins) et no-holo (copy without holo), feature (red ratio) : 20 vidéos au total , prédiction:100%
![ratio_red_classification_regression](https://github.com/user-attachments/assets/11847a2e-005c-491c-ab2f-60b93cf7866f)

holo (origins) et no-holo (photo_replacement), feature (green ratio) : 20 vidéos au total , prédiction:84,21% 
![ratio_green_avec photo_replacement](https://github.com/user-attachments/assets/c35daf6d-1c6f-428d-8bbd-3cfa21b25ff1)

**Conclusion :**
L’approche mise en place a donné de bons résultats pour détecter et classer les hologrammes présents sur les passeports. En utilisant des mosaïques générées à partir de patchs, nous avons pu distinguer efficacement les passeports contenant un hologramme de ceux qui n’en ont pas.  

**Pistes d’amélioration :**
Pour aller plus loin, une des perspectives serait d’adapter notre méthode à des passeports présentant des hologrammes plus complexes ou moins visibles, également tester de vrais passeports avec différents motifs d'hologrammes. Il serait aussi intéressant d’exploiter la dimension temporelle des vidéos en utilisant un réseau de neurones convolutif 3D (CNN 3D), ce qui permettrait de mieux capter les variations lumineuses liées aux hologrammes et d'améliorer la précision de la classification. 

**Projet réalisé par : BENMEHREZ Dima Sabrine**
