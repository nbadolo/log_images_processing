from pathlib import Path
import re


table_path = '/home/nbadolo/Bureau/Aymard/These/CDS-Tables/maps_dat'

input_file = f"{table_path}/raw/ReadMe.txt"
output_file = f"{table_path}/clean/ReadMe.txt"

# Lecture en mode brut
with open(input_file, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

cleaned_lines = []

for line in lines:
    # 1. Remplacer tabulations par espaces
    line = line.replace("\t", " ")

    # 2. Supprimer caractères non ASCII invisibles
    line = re.sub(r"[^\x20-\x7E\n]", "", line)

    # 3. Supprimer espaces multiples
    line = re.sub(r" +", " ", line)

    # 4. Retirer espaces début/fin
    line = line.rstrip()

    cleaned_lines.append(line)

# Sauvegarde propre
with open(output_file, "w", encoding="ascii", errors="ignore") as f:
    for line in cleaned_lines:
        f.write(line + "\n")

print("Clean done ->", output_file)