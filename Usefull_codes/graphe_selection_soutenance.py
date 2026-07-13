import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

base = '/home/nbadolo/Bureau/Aymard/Tables/ML/Hipparcos/Hip'
files = {
    'Hip_new (McD parent)': 'Hip_new.csv',
    'Sample 45 (enriched ML)': 'Sample_hip_enriched_for_ML.csv'
}

plt.figure(figsize=(8,6))
import os
import pandas as pd
import matplotlib.pyplot as plt


base = '/home/nbadolo/Bureau/Aymard/Tables/ML/Hipparcos/Hip'
parent_fname = 'Hip_new.csv'
enriched_fname = 'Sample_hip_enriched_for_ML.csv'

# Read files
parent_path = os.path.join(base, parent_fname)
enriched_path = os.path.join(base, enriched_fname)
if not os.path.exists(parent_path):
    print(f'Fichier manquant: {parent_path}')
    raise SystemExit(1)
if not os.path.exists(enriched_path):
    print(f'Fichier manquant: {enriched_path}')
    raise SystemExit(1)

df_parent = pd.read_csv(parent_path)
df_enriched = pd.read_csv(enriched_path)

required = {'E_IR', 'Mv'}
for name, df in [('parent', df_parent), ('enriched', df_enriched)]:
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f'Colonnes manquantes dans {name} table: {missing}')
        raise SystemExit(1)

# Keep only rows with positive E_IR for plotting (log scale)
df_parent = df_parent.dropna(subset=['Mv', 'E_IR'])
df_enriched = df_enriched.dropna(subset=['Mv', 'E_IR'])
df_parent = df_parent[df_parent['E_IR'] > 0]
df_enriched = df_enriched[df_enriched['E_IR'] > 0]

# Try to detect a name column to identify duplicates (flexible)
name_candidates = ['HIP', 'hip', 'HIP_ID', 'Name', 'name', 'object', 'Object', 'ID', 'Id', 'star', 'Star']
name_col = None
for c in name_candidates:
    if c in df_parent.columns and c in df_enriched.columns:
        name_col = c
        break

if name_col is not None:
    parent_keys = df_parent[name_col].astype(str).str.strip().str.lower()
    enriched_keys = df_enriched[name_col].astype(str).str.strip().str.lower()
else:
    # fallback: create a composite key from rounded numeric values
    parent_keys = (df_parent['Mv'].round(3).astype(str) + '_' + df_parent['E_IR'].round(5).astype(str))
    enriched_keys = (df_enriched['Mv'].round(3).astype(str) + '_' + df_enriched['E_IR'].round(5).astype(str))

# Select parent-only stars (exclude those present in enriched)
enriched_set = set(enriched_keys)
if name_col is not None:
    mask_parent_only = ~parent_keys.isin(enriched_set)
else:
    mask_parent_only = ~parent_keys.isin(enriched_set)

parent_only = df_parent[mask_parent_only]

# Plotting
plt.figure(figsize=(8,6))
ax = plt.gca()

# Parent sample: subdued grey circles
# Prefer column 'L' if available (user-provided luminosity), then other common names
brightness_candidates = ['L', 'V', 'Vmag', 'vmag', 'V_MAG', 'V_mag', 'Vmag_ap', 'Vmag_best', 'Vmag_mean']
brightness_col = None
for c in brightness_candidates:
    if (c in df_parent.columns) or (c in df_enriched.columns):
        brightness_col = c
        break
if brightness_col is None:
    # No brightness column found: fallback to previous plotting
    if len(parent_only) > 0:
        ax.scatter(parent_only['Mv'], parent_only['E_IR'], label='McDonald et al. 2012, 2017', marker='o',
                   facecolor='lightgray', edgecolor='k', s=50, alpha=0.35, linewidth=0.5, zorder=2)
    if len(df_enriched) > 0:
        ax.scatter(df_enriched['Mv'], df_enriched['E_IR'], label="échantillon d'étude", marker='*',
                   facecolor='cyan', edgecolor='k', s=220, alpha=1.0, linewidth=1.2, zorder=3)
else:
    # Color points by brightness column (if present). For magnitudes, lower=brighter.
    cmap = plt.cm.plasma_r
    # Parent: separate those with and without brightness values
    parent_with_b = parent_only[parent_only[brightness_col].notna()]
    parent_no_b = parent_only[parent_only[brightness_col].isna()]
    enriched_with_b = df_enriched[df_enriched[brightness_col].notna()]
    enriched_no_b = df_enriched[df_enriched[brightness_col].isna()]

    all_b_values = pd.concat([parent_with_b[brightness_col], enriched_with_b[brightness_col]]) if (len(parent_with_b) + len(enriched_with_b))>0 else None
    if all_b_values is not None and len(all_b_values) > 0:
        vmin = float(all_b_values.min())
        vmax = float(all_b_values.max())
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    if len(parent_no_b) > 0:
        ax.scatter(parent_no_b['Mv'], parent_no_b['E_IR'], label='McDonald et al. 2012, 2017', marker='o',
                   facecolor='lightgray', edgecolor='k', s=50, alpha=0.35, linewidth=0.5, zorder=2)
    if len(parent_with_b) > 0:
        lab_parent = 'McDonald et al. 2012, 2017' if len(parent_no_b) == 0 else None
        sc1 = ax.scatter(parent_with_b['Mv'], parent_with_b['E_IR'], c=parent_with_b[brightness_col], cmap=cmap, norm=norm,
                         marker='o', edgecolor='k', s=60, alpha=0.35, linewidth=0.5, zorder=2, label=lab_parent)

    # Enriched: prefer showing only enriched marker if duplicate
    if len(enriched_no_b) > 0:
        sc2 = ax.scatter(enriched_no_b['Mv'], enriched_no_b['E_IR'], label="échantillon d'étude", marker='*',
                         facecolor='cyan', edgecolor='k', s=220, alpha=1.0, linewidth=1.2, zorder=3)
    if len(enriched_with_b) > 0:
        lab_enriched = "échantillon d'étude" if len(enriched_no_b) == 0 else None
        sc3 = ax.scatter(enriched_with_b['Mv'], enriched_with_b['E_IR'], c=enriched_with_b[brightness_col], cmap=cmap, norm=norm,
                         marker='*', edgecolor='k', s=220, alpha=1.0, linewidth=1.2, zorder=3, label=lab_enriched)

    # Add colorbar if we plotted colored points
    try:
            if norm is not None:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                # Place colorbar on top (horizontal)
                # Place colorbar above the axes, outside the plot area
                cbar = plt.colorbar(sm, ax=ax, pad=0.06, orientation='horizontal', location='top')
                # Label depends on whether this is a magnitude-like column
                if 'mag' in brightness_col.lower() or brightness_col.lower() in ['v', 'vmag']:
                    cbar.set_label('V (mag)')
                else:
                    cbar.set_label(brightness_col)
                # Make colorbar label bold and readable for presentation
                try:
                    cbar.ax.xaxis.label.set_fontsize(16)
                    cbar.ax.xaxis.label.set_fontweight('bold')
                    # Put label and ticks on top of the colorbar
                    try:
                        cbar.ax.xaxis.set_label_position('top')
                        cbar.ax.xaxis.tick_top()
                    except Exception:
                        pass
                except Exception:
                    pass
                # Format colorbar ticks in scientific power notation and set bold
                try:
                    from matplotlib.ticker import ScalarFormatter
                    fmt = ScalarFormatter(useMathText=True)
                    fmt.set_powerlimits((-3, 3))
                    cbar.ax.xaxis.set_major_formatter(fmt)
                    # Bold and enlarge tick labels
                    for lbl in cbar.ax.get_xticklabels():
                        lbl.set_fontweight('bold')
                        lbl.set_fontsize(16)
                    # Also bold the offset text (the 10^n)
                    off = cbar.ax.xaxis.get_offset_text()
                    off.set_fontweight('bold')
                    off.set_fontsize(16)
                except Exception:
                    pass
    except Exception:
        pass

ax.set_xlabel('$\mathbf{M_R}$', fontsize=16, fontweight='bold')
ax.set_ylabel("E_IR (échelle log)", fontsize=16, fontweight='bold')
#ax.set_title('E_IR vs Mv', fontsize=16, fontweight='bold')
ax.set_yscale('log')

# Ticks formatting: fontsize 16 and bold for both axes
ax.tick_params(axis='both', which='major', labelsize=16)
for lbl in ax.get_xticklabels(which='major'):
    lbl.set_fontweight('bold')
for lbl in ax.get_yticklabels(which='major'):
    lbl.set_fontweight('bold')

# Create legend with symbol shapes only (no color influence from colormap)
handles = []
labels = []
# Parent sample handle (circle)
handles.append(Line2D([0], [0], marker='o', color='k', markerfacecolor='none', markeredgecolor='k', markersize=8, linestyle='None'))
labels.append('McDonald et al. 2012, 2017')
# Enriched sample handle (star)
handles.append(Line2D([0], [0], marker='*', color='k', markerfacecolor='none', markeredgecolor='k', markersize=14, linestyle='None'))
labels.append("échantillon d'étude")
ax.legend(handles, labels, fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
# Increase top margin to ensure colorbar above figure isn't clipped
try:
    plt.subplots_adjust(top=0.88)
except Exception:
    pass
# Ensure presentation output directory exists and save file with 'L' in the name
output_dir = '/home/nbadolo/Bureau/Aymard/Presentation_Charts'
os.makedirs(output_dir, exist_ok=True)
out = os.path.join(output_dir, 'EIR_vs_Mv_L.png')
# Vertical guide line at V=9 (dashed) to indicate AO optimal limit — bright red
ax.axvline(9, linestyle='--', color='#FF0000', linewidth=2, zorder=1)
plt.savefig(out, dpi=200)
print('Saved:', out)
# Also save as PDF for presentation-quality output
out_pdf = out.replace('.png', '.pdf')
try:
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    print('Saved:', out_pdf)
except Exception:
    pass
plt.show()
