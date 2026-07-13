#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génération d'un camembert pour la classification des enveloppes stellaires
Pour présentation
"""

import os
import matplotlib.pyplot as plt

# Données de classification
resolved_count = 16       # Clairement résolues
marginal_count = 11       # Marginalement résolues
unresolved_count = 18     # Non résolues

total_classified = resolved_count + marginal_count + unresolved_count
print(f"Total des étoiles classées : {total_classified}")

# Dossier de sauvegarde
output_folder = '/home/nbadolo/Bureau/Aymard/Presentation_Charts/'
os.makedirs(output_folder, exist_ok=True)

# 📊 Camembert stylisé pour présentation PowerPoint
if total_classified > 0:
    labels = ['Clairement\nrésolues', 'Marginalement\nrésolues', 'Non\nrésolues']
    sizes = [resolved_count, marginal_count, unresolved_count]
    # Palette professionnelle et contrastée
    colors = ['#1E3FA6', '#6F89C6', '#B7C4D8']
  # Vert, Orange, Rouge
    explode = (0.02, 0.02, 0.02)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='none')  # Fond transparent
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f'{int(pct/100*sum(sizes))}\n({pct:.1f}%)',
        startangle=140,
        explode=explode,
        textprops={'fontsize': 15, 'weight': 'bold'},
        wedgeprops={'linewidth': 2.5, 'edgecolor': 'white'},
        shadow=False  # Ombre pour profondeur
    )
    
    # Style des labels (noms des catégories)
    plt.setp(texts, fontsize=15, weight='bold', color='#3A3A3A')
    
    # Style des pourcentages/nombres
    plt.setp(autotexts, fontsize=14, weight='bold', color='white')
    
    ax.axis('equal')

    # Sauvegarder PNG et PDF (fond transparent pour PowerPoint)
    pie_path_png = os.path.join(output_folder, 'envelope_classification_pie.png')
    pie_path_pdf = os.path.join(output_folder, 'envelope_classification_pie.pdf')
    
    plt.savefig(pie_path_png, dpi=300, bbox_inches='tight', transparent=True, facecolor='none')
    plt.savefig(pie_path_pdf, dpi=300, bbox_inches='tight', transparent=True, facecolor='none')
    
    print(f"Camembert sauvegardé dans :")
    print(f"  - PNG : {pie_path_png}")
    print(f"  - PDF : {pie_path_pdf}")
    
    plt.show()
else:
    print("Aucune étoile classifiée, camembert non généré.")
