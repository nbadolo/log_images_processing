import re
import pandas as pd


tex_dist_path = '/home/nbadolo/Bureau/Aymard/Tables/ML/Hipparcos/Hip/'
def tex_to_csv(tex_path, csv_path):
    """
    Convertit une table LaTeX (.tex) en fichier CSV.
    - Ignore les lignes de commande LaTeX et les commentaires.
    - Suppose que les données sont séparées par '&' et chaque ligne se termine par '\\'.
    """
    with open(tex_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        # Ignore les commandes LaTeX et les commentaires
        if not line or line.startswith('%') or 'toprule' in line or 'midrule' in line or 'bottomrule' in line:
            continue
        # Nettoyage et extraction des données
        parts = [p.strip().replace('$', '').replace('\\', '').replace('\quad', '').replace('\rm', '').replace('_', '').replace('{', '').replace('}', '') for p in line.split('&')]
        # Ignore les lignes trop courtes
        if len(parts) < 2:
            continue
        # Retire le '\\' final si présent
        if parts[-1].endswith('\\'):
            parts[-1] = parts[-1][:-2].strip()
        data.append(parts)
    # Création du DataFrame
    df = pd.DataFrame(data)
    # Sauvegarde en CSV
    df.to_csv(csv_path, index=False, header=False)
    print(f"✅ Table convertie : {csv_path}")

# Exemple d'utilisation :
tex_to_csv(tex_dist_path + 'Sample_summary_table_hip_styled.tex', tex_dist_path + 'Sample_summary_table_hip_styled.csv')
