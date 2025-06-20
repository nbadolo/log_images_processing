#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 18:12:05 2025

@author: nbadolo
"""

from PIL import Image, ImageOps
import os
import math
"""
Ce code permet de fusionner plusieurs images pour un paipier A&A. Il les range 02 images
par ligne et 7 lignes par page, ie. 14 images par page. Il genere autant de pages qu'il faut. 
Mieux que son grand frère colla_image_for_paper_pro, i place preferntiellement en bas de page
les images qui ont x_label et un y_label.
Les dossiers input et output sont dans le dossier 
"/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/collage/"
"""

# === CONFIGURATION ===
folder_name = "Pol_Int"
psf_or_no = "no_psf"
#psf_or_no = "no_psf"
input_folder = "/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/collage/" + folder_name + "/" + psf_or_no + "/leaf1"
output_folder = "/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/collage/output/" + folder_name + "/" +  psf_or_no + "/leaf1"
os.makedirs(output_folder, exist_ok=True)

page_width, page_height = 2480, 3508  # A4 en pixels à 300dpi
images_per_row = 2
rows_per_page = 7
images_per_page = images_per_row * rows_per_page  # 14 images max par page
thumb_size = (page_width // images_per_row, page_height // rows_per_page)

# === DÉTECTION DES LABELS (X et Y labels) ===
def has_xlabel_ylabel(img, threshold=10):
    """
    Cette fonction vérifie si une image contient un xlabel et un ylabel
    en scannant les zones du bas et à gauche de l'image.
    """
    grayscale = ImageOps.grayscale(img)
    w, h = grayscale.size
    bottom = grayscale.crop((0, int(h * 0.85), w, h))  # Bas de l'image
    left = grayscale.crop((0, 0, int(w * 0.15), h))  # Côté gauche
    dark_bottom = sum(1 for p in bottom.getdata() if p < 240)
    dark_left = sum(1 for p in left.getdata() if p < 240)
    return dark_bottom > threshold and dark_left > threshold

# === REDIMENSIONNE LES IMAGES ET LES CENTRE SUR UN FOND BLANC ===
def fit_image(img, size):
    """
    Cette fonction redimensionne l'image tout en conservant ses proportions
    et la centre sur un fond blanc.
    """
    img.thumbnail(size)
    bg = Image.new("RGB", size, (255, 255, 255))
    offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
    bg.paste(img, offset)
    return bg

# === CHARGEMENT DES IMAGES ===
image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
images_with_labels = []
images_without_labels = []

# Classification des images
for fname in image_files:
    try:
        path = os.path.join(input_folder, fname)
        img = Image.open(path)
        resized = fit_image(img, thumb_size)
        if has_xlabel_ylabel(resized):  # Image avec xlabel et ylabel
            images_with_labels.append(resized)
        else:
            images_without_labels.append(resized)  # Image sans xlabel ni ylabel
    except Exception as e:
        print(f"Erreur avec {fname} : {e}")

# === ORGANISATION DES IMAGES EN PAGES ===
all_images = []

# Remplir les pages avec des images sans labels (maximum 14 par page)
while len(images_without_labels) >= (images_per_page - len(images_with_labels) % images_per_row):
    page_imgs = [images_without_labels.pop(0) for _ in range(images_per_page - len(images_with_labels) % images_per_row)]
    all_images.extend(page_imgs)

# Ajouter les images avec labels à la fin de chaque page
for i in range(0, len(images_with_labels), images_per_page):
    page_images = images_with_labels[i:i + images_per_page]  # Remplir la page en priorité
    all_images.extend(page_images)

# === CRÉATION DES COLLAGES ===
page_num = 1
for i in range(0, len(all_images), images_per_page):
    page_images = all_images[i:i + images_per_page]
    collage = Image.new('RGB', (page_width, page_height), color=(255, 255, 255))

    for idx, img in enumerate(page_images):
        row = idx // images_per_row
        col = idx % images_per_row
        x = col * thumb_size[0]
        y = row * thumb_size[1]
        collage.paste(img, (x, y))

    # Enregistrer chaque page avec un nom dynamique
    collage_path = os.path.join(output_folder, f"{folder_name}_{page_num:02d}.jpg")
    collage.save(collage_path)
    print(f"✅ Page {page_num} enregistrée : {collage_path}")
    page_num += 1

# === FIN ===
print("\n🎉 Toutes les pages ont été générées correctement.")
