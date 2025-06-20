#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 17:17:24 2025

@author: nbadolo
"""

from PIL import Image
import os
import math

# === Config ===
folder_name = "Pol_Int"
input_folder = "/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/collage/" +folder_name+"/"
output_folder = "/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/collage/"+folder_name+"/"
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
image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
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
    collage_path = os.path.join(output_folder, f"page_{page+1:02d}.jpg")
    collage.save(collage_path)
    print(f"Page {page+1} sauvegardée : {collage_path}")
