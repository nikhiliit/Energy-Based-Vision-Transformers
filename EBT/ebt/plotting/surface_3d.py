import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import proj3d
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from PIL import Image as PILImage

# Styling
BEIGE_BG = "#FAF9F6"
SURFACE_CMAP = LinearSegmentedColormap.from_list(
    "minimal_basin",
    [(0.00, "#1B2A5C"), (0.15, "#3C5A9A"), (0.30, "#7B9FCC"),
     (0.45, "#C8D5E3"), (0.52, "#FAF9F6"), (0.60, "#E8C8B0"),
     (0.72, "#D4916A"), (0.85, "#C04828"), (1.00, "#8B2010")]
)
SAMPLE_A_COLOR = "#FF2D55"
SAMPLE_B_COLOR = "#00C8FF"
SAMPLE_A_TRAIL = "#FF8C9A"
SAMPLE_B_TRAIL = "#80E8FF"


def build_energy_surface(feats_2d, energies_class, grid_bounds, res=100, sigma=3.0, smooth=5.0):
    x_min, x_max, y_min, y_max = grid_bounds
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))
    
    plateau = np.mean(energies_class) + 0.5 * np.std(energies_class)
    bpts = []
    for v in np.linspace(x_min, x_max, 20):
        bpts.extend([[v, y_min], [v, y_max]])
    for v in np.linspace(y_min, y_max, 20):
        bpts.extend([[x_min, v], [x_max, v]])
    for cx in [x_min, x_max]:
        for cy in [y_min, y_max]:
            bpts.append([cx, cy])
    
    pts = np.vstack([feats_2d, bpts])
    vals = np.concatenate([energies_class, np.full(len(bpts), plateau)])
    rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=smooth)
    eg = rbf(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    eg = np.clip(eg, energies_class.min() - 0.3 * np.std(energies_class), plateau + 0.5 * np.std(energies_class))
    
    if sigma > 0:
        eg = gaussian_filter(eg, sigma=sigma)
    return xx, yy, eg


def _get_z(x, y, xx, yy, eg):
    interp = RegularGridInterpolator((xx[0, :], yy[:, 0]), eg.T, method="linear", bounds_error=False, fill_value=None)
    return float(interp([[x, y]]))


def _paint_surface(ax, xx, yy, eg, e_min, e_max):
    norm = plt.Normalize(vmin=e_min, vmax=e_max)
    ls = LightSource(azdeg=315, altdeg=45)
    ax.plot_surface(xx, yy, eg, facecolors=SURFACE_CMAP(norm(eg)), alpha=0.93, shade=True, lightsource=ls, antialiased=True)
    ax.plot_wireframe(xx, yy, eg, rstride=8, cstride=8, color="white", alpha=0.04, linewidth=0.2)


def _paint_sample(ax, pos, pil, color, trail_color, past, xx, yy, eg):
    if past:
        for i, (tx, ty) in enumerate(past):
            tz = _get_z(tx, ty, xx, yy, eg)
            ax.scatter([tx], [ty], [tz], c=trail_color, s=30 * (i/len(past)), alpha=0.5 * (i/len(past)))
    
    sx, sy = pos
    sz = _get_z(sx, sy, xx, yy, eg)
    ax.scatter([sx], [sy], [sz], c=color, s=55, alpha=0.8, zorder=11)
    
    if pil:
        thumb = pil.copy(); thumb.thumbnail((56, 56))
        x2, y2, _ = proj3d.proj_transform(sx, sy, sz, ax.get_proj())
        ax.add_artist(AnnotationBbox(OffsetImage(thumb, zoom=0.5), (x2, y2), frameon=False, xycoords=ax.transData))


def render_3d_frame(style="minimal", **kwargs):
    """
    Unified 3D frame renderer.
    Styles: 'minimal', 'sample', 'dual'
    """
    fig = plt.figure(figsize=(14 if style == "dual" else 7, 7), facecolor=BEIGE_BG)
    elev, azim = kwargs.get("elev", 28), kwargs.get("azim", -55)

    if style != "dual":
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        ax.set_facecolor(BEIGE_BG)
        xx, yy, eg = kwargs["xx"], kwargs["yy"], kwargs["eg"]
        e_min, e_max = kwargs.get("e_range", (eg.min(), eg.max()))
        _paint_surface(ax, xx, yy, eg, e_min, e_max)
        
        if style == "sample":
            _paint_sample(ax, kwargs["pos"], kwargs.get("pil"), kwargs["color"], kwargs.get("trail_color"), kwargs.get("past"), xx, yy, eg)
        
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
    else:
        # Dual Panel Logic (simplified)
        for i in [121, 122]:
            ax = fig.add_subplot(i, projection="3d", computed_zorder=False)
            ax.set_facecolor(BEIGE_BG)
            k = "a" if i == 121 else "b"
            xx, yy, eg = kwargs[f"xx_{k}"], kwargs[f"yy_{k}"], kwargs[f"eg_{k}"]
            e_min, e_max = kwargs.get(f"e_range_{k}", (eg.min(), eg.max()))
            _paint_surface(ax, xx, yy, eg, e_min, e_max)
            # Add samples to both panels
            _paint_sample(ax, kwargs["pos_a"], kwargs.get("pil_a"), SAMPLE_A_COLOR, SAMPLE_A_TRAIL, kwargs.get("past_a"), xx, yy, eg)
            _paint_sample(ax, kwargs["pos_b"], kwargs.get("pil_b"), SAMPLE_B_COLOR, SAMPLE_B_TRAIL, kwargs.get("past_b"), xx, yy, eg)
            ax.set_axis_off()
            ax.view_init(elev=elev, azim=azim)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return PILImage.fromarray(rgba).convert("RGB")
