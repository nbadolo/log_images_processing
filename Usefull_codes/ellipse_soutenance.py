import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from astropy.io import fits
from scipy import ndimage

# -----------------------------
# Utils: ellipse fit from mask
# -----------------------------
def fit_ellipse_from_mask(mask: np.ndarray):
    """
    Fit an ellipse to a binary mask using second moments (PCA on pixel coords).
    Returns (xc, yc, a, b, theta) where:
      - (xc, yc) center in pixel coordinates
      - a, b semi-axes (pixels), with a >= b
      - theta angle in radians (matplotlib convention: CCW from x-axis)
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None

    xc = xs.mean()
    yc = ys.mean()

    X = np.column_stack([xs - xc, ys - yc])
    # covariance
    C = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # semi-axes: scale factor controls how "wide" the ellipse is
    # For a uniform filled ellipse, moments relate to axes; here we use a pragmatic scale.
    scale = 1.8
    a = scale * np.sqrt(max(eigvals[0], 1e-12))
    b = scale * np.sqrt(max(eigvals[1], 1e-12))

    vx, vy = eigvecs[:, 0]
    theta = np.arctan2(vy, vx)
    return xc, yc, a, b, theta


def largest_connected_component(mask: np.ndarray):
    """Keep only the largest connected component (8-connectivity)."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    biggest = 1 + int(np.argmax(sizes))
    return lab == biggest


# -----------------------------
# Main: build figure
# -----------------------------
def make_threshold_figure(
    fits_path: str,
    h_list=(0.005, 0.02, 0.05, 0.10),  # 0.5%, 2%, 5%, 10%
    output_png="thresholding_PI_L.png",
    crop=None,  # crop=(ymin,ymax,xmin,xmax) if you want
    add_ellipse=True,
    transparent=True
):
    # Read FITS (assumes PI_L in primary HDU; adapt if needed)
    img = fits.getdata(fits_path)
    img = np.array(img, dtype=float)

    # Clean NaNs/Infs
    img[~np.isfinite(img)] = np.nan

    # Optional crop
    if crop is not None:
        ymin, ymax, xmin, xmax = crop
        img = img[ymin:ymax, xmin:xmax]

    # Use max ignoring NaNs
    pi_max = np.nanmax(img)
    vmin = np.nanmin(img)
    vmax = np.nanmax(img)
    if not np.isfinite(pi_max) or pi_max <= 0:
        raise ValueError("PI_L,max invalid (<=0 or NaN). Check your image.")

    # Figure layout: 1 + len(h_list) panels
    n_masks = len(h_list)
    ncols = min(3, 1 + n_masks)  # up to 3 columns for readability
    nrows = int(np.ceil((1 + n_masks) / ncols))

    fig = plt.figure(figsize=(4.2 * ncols, 3.6 * nrows))
    fig.patch.set_alpha(0.0 if transparent else 1.0)

    # Panel 1: continuous PI_L
    ax0 = fig.add_subplot(nrows, ncols, 1)
    vmin = np.nanmin(img)
    vmax = np.nanmax(img)
    im0 = ax0.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap="inferno")
    ax0.set_title(r"$\mathrm{PI}_L(x,y)$")
    ax0.set_axis_off()
    # Pas de colorbar pour garder un centrage parfait

    # Mask panels
    for i, h in enumerate(h_list, start=2):
        ax = fig.add_subplot(nrows, ncols, i)

        thr = h * pi_max
        mask = np.zeros_like(img, dtype=bool)
        mask[np.isfinite(img)] = img[np.isfinite(img)] > thr

        # keep only largest region (optional but usually helps for envelopes)
        mask_main = largest_connected_component(mask)

        ax.imshow(mask_main.astype(float), origin="lower")  # default cmap is fine
        ax.set_title(rf"$h={100*h:.1f}\% \times \mathrm{{PI}}_{{L,\max}}$")
        ax.set_axis_off()

        # Optional ellipse overlay
        if add_ellipse:
            params = fit_ellipse_from_mask(mask_main)
            if params is not None:
                xc, yc, a, b, theta = params
                t = np.linspace(0, 2*np.pi, 400)
                # parametric ellipse
                x = xc + a*np.cos(t)*np.cos(theta) - b*np.sin(t)*np.sin(theta)
                y = yc + a*np.cos(t)*np.sin(theta) + b*np.sin(t)*np.cos(theta)
                ax.plot(x, y, linewidth=2)  # default line color

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0, transparent=transparent)
    plt.show()
    print(f"Saved: {output_png}")


def make_threshold_figure_synthetic(
    h_list=(0.0005, 0.05, 0.10),
    output_png="thresholding_PI_L_synthetic.png",
    shape=(300, 300),
    center=(150, 150),
    sigma_x=40,
    sigma_y=20,
    theta_deg=30,
    amplitude=1.0,
    add_ellipse=True,
    transparent=True,
    layout="triangle"
):
    """
    Crée une figure de seuils sur une gaussienne 2D elliptique synthétique.
    Utile pour l'illustration en présentation (sans FITS réel).
    """
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    xc, yc = center
    theta = np.deg2rad(theta_deg)

    x0 = x - xc
    y0 = y - yc

    # Rotation
    xr = x0 * np.cos(theta) + y0 * np.sin(theta)
    yr = -x0 * np.sin(theta) + y0 * np.cos(theta)

    img = amplitude * np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))

    # Reuse the same logic as make_threshold_figure (inline)
    img = np.array(img, dtype=float)
    img[~np.isfinite(img)] = np.nan

    pi_max = np.nanmax(img)
    vmin = np.nanmin(img)
    vmax = np.nanmax(img)
    if not np.isfinite(pi_max) or pi_max <= 0:
        raise ValueError("PI_L,max invalid (<=0 or NaN). Check your image.")

    # Style présentation
    plt.rcParams.update({
        "font.size": 14,
        "font.family": "DejaVu Sans",
    })

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_alpha(0.0 if transparent else 1.0)

    # Dispositions possibles
    if layout == "triangle":
        ax0_pos = [0.34, 0.34, 0.32, 0.32]
        panel_positions = [
            [0.34, 0.68, 0.32, 0.28],  # haut-centre
            [0.08, 0.08, 0.32, 0.28],  # bas-gauche
            [0.60, 0.08, 0.32, 0.28],  # bas-droit
        ]
        fig.set_size_inches(10, 8)
    elif layout == "grid2x2":
        ax0_pos = [0.08, 0.55, 0.38, 0.38]
        panel_positions = [
            [0.54, 0.55, 0.38, 0.38],  # haut-droit
            [0.08, 0.08, 0.38, 0.38],  # bas-gauche
            [0.54, 0.08, 0.38, 0.38],  # bas-droit
        ]
        fig.set_size_inches(10, 8)
    elif layout == "horizontal":
        ax0_pos = [0.02, 0.18, 0.24, 0.64]
        panel_positions = [
            [0.28, 0.18, 0.24, 0.64],
            [0.54, 0.18, 0.24, 0.64],
            [0.80, 0.18, 0.18, 0.64],
        ]
        fig.set_size_inches(14, 5)
    elif layout == "top1_bottom3":
        ax0_pos = [0.31, 0.54, 0.38, 0.36]
        panel_positions = [
            [0.05, 0.12, 0.28, 0.38],
            [0.36, 0.12, 0.28, 0.38],
            [0.67, 0.12, 0.28, 0.38],
        ]
        fig.set_size_inches(10, 8)
    else:
        raise ValueError("layout inconnu: choisir triangle, grid2x2, horizontal, top1_bottom3")

    # Panneau central
    ax0 = fig.add_axes(ax0_pos)
    im0 = ax0.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap="inferno")
    ax0.text(
        0.5,
        0.92,
        r"$\mathrm{PI}_L(x,y)$ (étoile)",
        transform=ax0.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.5, linewidth=0),
    )
    ax0.set_axis_off()

    # Pré-calcul des ellipses par seuil
    params_by_h = {}
    for h in h_list:
        thr = h * pi_max
        mask = np.zeros_like(img, dtype=bool)
        mask[np.isfinite(img)] = img[np.isfinite(img)] > thr
        mask_main = largest_connected_component(mask)
        params_by_h[h] = fit_ellipse_from_mask(mask_main)

    # Remapping des tailles selon la demande:
    # 0.5% <- taille de 5%
    # 5%   <- taille de 10%
    # 10%  <- taille de 10% mais plus petite (facteur)
    size_map = {
        0.0005: 0.05,
        0.05: 0.10,
        0.10: 0.10,
    }
    shrink_10 = 0.75  # facteur de réduction supplémentaire pour 10%

    for i, h in enumerate(h_list):
        ax = fig.add_axes(panel_positions[i])

        # Afficher la même image PI_L (tache jaune constante)
        ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap="inferno")
        if np.isclose(h, 0.0005):
            label_text = r"$h=0.5\% \times \mathrm{PI}_{L,\max}$"
        else:
            label_text = rf"$h={100*h:.0f}\% \times \mathrm{{PI}}_{{L,\max}}$"
        if np.isclose(h, 0.05):
            label_text = r"$\hat{h}$"
        ax.text(
            0.5,
            0.92,
            label_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=14,
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.5, linewidth=0),
        )
        ax.set_axis_off()

        if add_ellipse:
            h_src = size_map.get(h, h)
            params = params_by_h.get(h_src)
            if params is not None:
                xc, yc, a, b, theta = params
                if h == 0.10:
                    a *= shrink_10
                    b *= shrink_10
                t = np.linspace(0, 2*np.pi, 400)
                x = xc + a*np.cos(t)*np.cos(theta) - b*np.sin(t)*np.sin(theta)
                y = yc + a*np.cos(t)*np.sin(theta) + b*np.sin(t)*np.cos(theta)
                ax.plot(
                    x,
                    y,
                    color="#00BFFF",
                    linewidth=3.0,
                    path_effects=[
                        pe.Stroke(linewidth=4.5, foreground="black"),
                        pe.Normal(),
                    ],
                )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight", transparent=transparent)
    plt.show()
    print(f"Saved: {output_png}")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Comparer plusieurs dispositions pour présentation
    base_out = "/home/nbadolo/Bureau/Aymard/Presentation_Charts/thresholding_PI_L_synthetic"
    for layout in ["triangle", "grid2x2", "horizontal", "top1_bottom3"]:
        out = f"{base_out}_{layout}.png"
        make_threshold_figure_synthetic(
            h_list=(0.0005, 0.05, 0.10),
            output_png=out,
            shape=(300, 300),
            center=(150, 150),
            sigma_x=45,
            sigma_y=20,
            theta_deg=35,
            amplitude=1.0,
            add_ellipse=True,
            transparent=True,
            layout=layout,
        )
