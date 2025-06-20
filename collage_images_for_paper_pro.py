#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 17:32:37 2025

@author: nbadolo
"""

from PIL import Image
import os
import math




"""
Ce code permet de fusionner plusieurs images pour un paipier A&A. Il les range 02 images
par ligne et 7 ligne par page, ie. 14 images par page. Il genere autant de pages qu'il faut. 
"""

# === Config ===
folder_name = "Pol_Int"
input_folder = "/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/collage/" + folder_name + "/"
output_folder = "/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/collage/output/" + folder_name + "/"
os.makedirs(output_folder, exist_ok=True)

page_width, page_height = 2480, 3508  # Format A4 portrait (pixels à 300dpi)
images_per_row = 2
rows_per_page = 7
images_per_page = images_per_row * rows_per_page
thumb_size = (page_width // images_per_row, page_height // rows_per_page)

def fit_image(img, size):
    img.thumbnail(size)
    bg = Image.new("RGB", size, (255, 255, 255))
    offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
    bg.paste(img, offset)
    return bg

# === Charger les images ===
image_files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
])
images = []

for f in image_files:
    try:
        img_path = os.path.join(input_folder, f)
        img = Image.open(img_path)
        img = fit_image(img, thumb_size)
        images.append(img)
    except Exception as e:
        print(f"Erreur avec {f} : {e}")

# === Créer les pages ===
num_pages = math.ceil(len(images) / images_per_page)
page_paths = []

for page in range(num_pages):
    collage = Image.new('RGB', (page_width, page_height), color=(255, 255, 255))
    for idx in range(images_per_page):
        global_index = page * images_per_page + idx
        if global_index >= len(images):
            break
        img = images[global_index]
        row = idx // images_per_row
        col = idx % images_per_row
        x = col * thumb_size[0]
        y = row * thumb_size[1]
        collage.paste(img, (x, y))
    
    # Nom du fichier avec folder_name
    collage_filename = f"{folder_name}_{page+1:02d}.jpg"
    collage_path = os.path.join(output_folder, collage_filename)
    collage.save(collage_path)
    page_paths.append(collage_path)
    print(f"Page {page+1} sauvegardée : {collage_path}")

# === Créer le PDF final ===
if page_paths:
    images_for_pdf = [Image.open(p).convert("RGB") for p in page_paths]
    pdf_output_path = os.path.join(output_folder, f"{folder_name}.pdf")
    images_for_pdf[0].save(pdf_output_path, save_all=True, append_images=images_for_pdf[1:], resolution=300)
    print(f"✅ PDF généré avec succès : {pdf_output_path}")

else:
    print("Aucune page à intégrer dans le PDF.")
