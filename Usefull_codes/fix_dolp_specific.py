import re

# Read the file
with open('polarized_maps_paper&_thesis.py', 'r') as f:
    content = f.read()

# Find and replace the DoLP_specific axes section
old_pattern = r"# Carte DoLP seule : PAS de labels ni graduations\s+ax_dolp\.set_xticks\(\[\]\)\s+ax_dolp\.set_yticks\(\[\]\)\s+ax_dolp\.tick_params\(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False\)"

new_text = """# Carte DoLP seule : avec labels et graduations
            ax_dolp.set_xlabel('Relative RA (mas)', fontsize=label_size)
            ax_dolp.set_ylabel('Relative Dec (mas)', fontsize=label_size)
            ax_dolp.tick_params(axis='both', labelsize=label_size, width=1.2)
            ax_dolp.locator_params(axis='x', nbins=5)
            ax_dolp.locator_params(axis='y', nbins=5)"""

content_new = re.sub(old_pattern, new_text, content)

# Write back
with open('polarized_maps_paper&_thesis.py', 'w') as f:
    f.write(content_new)

print("Done! DoLP_specific axes have been updated.")
