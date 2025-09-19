import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 12))
ax.set_xlim(-10, 10)
ax.set_ylim(-1, 15)
ax.set_aspect('equal')
ax.axis('off')

def add_arrowed_text(x, y, text, arrow_props=None):
    if arrow_props:
        ax.annotate(text, xy=(x, y), xytext=(x + 1, y), arrowprops=arrow_props, fontsize=10)
    else:
        ax.text(x, y, text, fontsize=10, ha='center')

# Cœur inerte
core = patches.Circle((0, 0), 1, facecolor='darkred', edgecolor='black', label='Cœur inerte (CO/O/Ne)')
ax.add_patch(core)
add_arrowed_text(0, 0, "Cœur inerte\n(CO/O/Ne/Mg)", arrow_props=dict(arrowstyle='->'))

# Couche de combustion de l'hélium (anneau)
he_shell_outer = patches.Circle((0, 0), 1.5, facecolor='orange', edgecolor='black', label='Couche He-burning')
he_shell_inner = patches.Circle((0, 0), 1, facecolor='white', edgecolor='none')
ax.add_patch(he_shell_outer)
ax.add_patch(he_shell_inner)
add_arrowed_text(0, 1.25, "Couche He-burning\n(Processus triple-alpha)", arrow_props=dict(arrowstyle='->'))

# Couche de combustion de l'hydrogène (anneau)
h_shell_outer = patches.Circle((0, 0), 2, facecolor='yellow', edgecolor='black', label='Couche H-burning')
h_shell_inner = patches.Circle((0, 0), 1.5, facecolor='white', edgecolor='none')
ax.add_patch(h_shell_outer)
ax.add_patch(h_shell_inner)
add_arrowed_text(0, 1.75, "Couche H-burning\n(Cycle CNO)", arrow_props=dict(arrowstyle='->'))

# Zone intercoquille (anneau)
inter_shell_outer = patches.Circle((0, 0), 2.5, facecolor='lightgray', edgecolor='black', label='Zone intercoquille')
inter_shell_inner = patches.Circle((0, 0), 2, facecolor='white', edgecolor='none')
ax.add_patch(inter_shell_outer)
ax.add_patch(inter_shell_inner)
add_arrowed_text(0, 2.25, "Zone intercoquille\n(Nucléosynthèse s)", arrow_props=dict(arrowstyle='->'))

# Enveloppe convective (anneau)
conv_env_outer = patches.Circle((0, 0), 4, facecolor='lightblue', edgecolor='black', label='Enveloppe convective')
conv_env_inner = patches.Circle((0, 0), 2.5, facecolor='white', edgecolor='none')
ax.add_patch(conv_env_outer)
ax.add_patch(conv_env_inner)
add_arrowed_text(0, 3, "Enveloppe convective\n(Dredge-up)", arrow_props=dict(arrowstyle='->'))

# Atmosphère stellaire (anneau)
atmosphere_outer = patches.Circle((0, 0), 5, facecolor='lightyellow', edgecolor='black', label='Atmosphère stellaire')
atmosphere_inner = patches.Circle((0, 0), 4, facecolor='white', edgecolor='none')
ax.add_patch(atmosphere_outer)
ax.add_patch(atmosphere_inner)
add_arrowed_text(0, 4.5, "Atmosphère stellaire\n(Formation de molécules)", arrow_props=dict(arrowstyle='->'))

# Enveloppe circumstellaire (anneau)
circum_env_outer = patches.Circle((0, 0), 8, facecolor='lightpink', edgecolor='black', alpha=0.5, label='Enveloppe circumstellaire')
circum_env_inner = patches.Circle((0, 0), 5, facecolor='white', edgecolor='none', alpha=0.5)
ax.add_patch(circum_env_outer)
ax.add_patch(circum_env_inner)
add_arrowed_text(0, 6.5, "Enveloppe circumstellaire\n(Poussières et gaz)", arrow_props=dict(arrowstyle='->'))

# Vent stellaire (arc)
wind = patches.Arc((0, 0), 16, 12, theta1=0, theta2=180, edgecolor='black', alpha=0.3, label='Vent stellaire')
ax.add_patch(wind)
add_arrowed_text(0, 9, "Vent stellaire\n(Perte de masse)", arrow_props=dict(arrowstyle='->'))

# Légende
legend_elements = [
    patches.Patch(facecolor='darkred', edgecolor='black', label='Cœur inerte'),
    patches.Patch(facecolor='orange', edgecolor='black', label='Couche He-burning'),
    patches.Patch(facecolor='yellow', edgecolor='black', label='Couche H-burning'),
    patches.Patch(facecolor='lightgray', edgecolor='black', label='Zone intercoquille'),
    patches.Patch(facecolor='lightblue', edgecolor='black', label='Enveloppe convective'),
    patches.Patch(facecolor='lightyellow', edgecolor='black', label='Atmosphère stellaire'),
    patches.Patch(facecolor='lightpink', edgecolor='black', alpha=0.5, label='Enveloppe circumstellaire'),
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)

plt.title("Structure d'une étoile AGB avec enveloppe circumstellaire", fontsize=14)
plt.tight_layout()
plt.show()