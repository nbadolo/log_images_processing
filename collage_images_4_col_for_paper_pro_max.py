#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 16:27:30 2025

@author: nbadolo
"""

"""
Créé le 22 avril 2025 par nbadolo.

Ce script assemble des images pour un article A&A. Il organise :
- 4 images par ligne,
- 7 lignes par page → 28 images max par page,
- Les images avec xlabel et ylabel sont toujours placées en bas de chaque page (dernière ligne).
"""

from PIL import Image, ImageOps
import os

# === CONFIGURATION ===
folder_name = "Pol_Int"
psf_or_no = "no_psf"
input_folder = f"/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/collage/{folder_name}/{psf_or_no}/leaf1"
output_folder = f"/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/collage/output/{folder_name}/{psf_or_no}/leaf1"
os.makedirs(output_folder, exist_ok=True)

# A4 dimensions in pixels (300 dpi)
page_width, page_height = 2480, 3508
images_per_row = 4
rows_per_page = 7
images_per_page = images_per_row * rows_per_page  # = 28
thumb_size = (page_width // images_per_row, page_height // rows_per_page)

# === FONCTION POUR DÉTECTER LA PRÉSENCE DE LABELS ===
def has_xlabel_ylabel(img, threshold=10):
    grayscale = ImageOps.grayscale(img)
    w, h = grayscale.size
    bottom = grayscale.crop((0, int(h * 0.85), w, h))
    left = grayscale.crop((0, 0, int(w * 0.15), h))
    dark_bottom = sum(1 for p in bottom.getdata() if p < 240)
    dark_left = sum(1 for p in left.getdata() if p < 240)
    return dark_bottom > threshold and dark_left > threshold

# === ADAPTATION DES IMAGES À LA GRILLE ===
def fit_image(img, size):
    img.thumbnail(size)
    bg = Image.new("RGB", size, (255, 255, 255))
    offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
    bg.paste(img, offset)
    return bg

# === CHARGEMENT DES IMAGES ===
image_files = sorted([f for f in os.listdir(input_folder)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
images_with_labels = []
images_without_labels = []

for fname in image_files:
    try:
        img_path = os.path.join(input_folder, fname)
        img = Image.open(img_path)
        resized = fit_image(img, thumb_size)
        if has_xlabel_ylabel(resized):
            images_with_labels.append(resized)
        else:
            images_without_labels.append(resized)
    except Exception as e:
        print(f"Erreur avec {fname} : {e}")

# === ORGANISATION DES IMAGES EN PAGES ===
all_pages = []

while True:
    if len(images_without_labels) >= (images_per_page - images_per_row) and len(images_with_labels) >= images_per_row:
        # Page complète : 24 sans label + 4 avec labels pour la dernière ligne
        page_imgs = [images_without_labels.pop(0) for _ in range(images_per_page - images_per_row)]
        page_imgs += [images_with_labels.pop(0) for _ in range(images_per_row)]
        all_pages.append(page_imgs)
    elif len(images_without_labels) + len(images_with_labels) >= images_per_page:
        # Remplissage hybride mais toujours une page pleine
        page_imgs = []
        while len(page_imgs) < (images_per_page - images_per_row) and images_without_labels:
            page_imgs.append(images_without_labels.pop(0))
        while len(page_imgs) < images_per_page and images_with_labels:
            page_imgs.append(images_with_labels.pop(0))
        while len(page_imgs) < images_per_page and images_without_labels:
            page_imgs.append(images_without_labels.pop(0))
        all_pages.append(page_imgs)
    else:
        # Dernière page incomplète
        remaining_imgs = images_without_labels + images_with_labels
        if remaining_imgs:
            all_pages.append(remaining_imgs)
        break

# === GÉNÉRATION DES COLLAGES ===
for page_num, page_images in enumerate(all_pages, start=1):
    collage = Image.new("RGB", (page_width, page_height), color=(255, 255, 255))

    for idx, img in enumerate(page_images):
        row = idx // images_per_row
        col = idx % images_per_row
        x = col * thumb_size[0]
        y = row * thumb_size[1]
        collage.paste(img, (x, y))

    out_path = os.path.join(output_folder, f"{folder_name}_{page_num:02d}.jpg")
    collage.save(out_path)
    print(f"✅ Page {page_num} enregistrée : {out_path}")

print("\n🎉 Collages générés avec succès.")
