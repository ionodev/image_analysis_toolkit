from __future__ import annotations
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import warnings
from PIL import Image
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 1. SETUP & DATA
mesh = pv.read("data/hydrogen.vtk")
print(mesh)
ref_intensities = mesh.active_scalars.copy()
# Normalize to [0, 1]
ref_intensities = (ref_intensities - ref_intensities.min()) / (ref_intensities.max() - ref_intensities.min())

# Basic visualization:
def plot_molecule():
    # Retrieve intensity values from the scalars in mesh
    ref_intensities = mesh.active_scalars.copy()

    # Normalize the intensity values to [0, 1]
    ref_intensities = (ref_intensities - ref_intensities.min()) / (ref_intensities.max() - ref_intensities.min())

    # Give the data points a reference to the intensity values
    mesh.point_data.clear()
    mesh.point_data["intensity"] = ref_intensities

    # Make an exponentially decreasing opacity for clearer visualization
    opacity_transfer = np.linspace(0, 1, 128) ** 1.3

    # Plot volume mesh
    pl = pv.Plotter()
    pl.add_volume(mesh, scalars="intensity", cmap='inferno', clim=[0, 1], opacity=opacity_transfer, show_scalar_bar=False)
    pl.add_scalar_bar(title="Intensity", vertical=True, n_labels=5, height=0.7, position_x=0.83, position_y=0.2)
    pl.view_zx()
    pl.show_grid()
    pl.show()

plot_molecule()

mesh.point_data.clear()
mesh.point_data["intensity"] = ref_intensities
h_mask_3d = ref_intensities >= 0.1

cmap='inferno'

# Mid-Z Plane Calculation
z_min, z_max = mesh.bounds[4], mesh.bounds[5]
z_mid = (z_min + z_max) / 2

opacity_transfer = np.linspace(0, 1, 128) ** 1.3

# 2. DATA GENERATION (Degraded Volumes)

# Parameters
salt_prob = 0.01 / 2
pepper_prob = 0.01 / 2
n_reconstructions = 27

cell_mesh = mesh.point_data_to_cell_data()
base_cell_data = cell_mesh.active_scalars.copy()

# Stacks for computations (point-based)
sp_stack_points, gauss_stack_points = [], []

# Stacks for 3D visualization (cell-based for S&P, point-based for Gaussian)
sp_stack_cells = []
sp_stack_cells_as_points = []

print(f"Generating {n_reconstructions} samples for cross-comparison...")

for i in range(n_reconstructions):
    # S&P noise (cell based)
    ni_cells = base_cell_data.copy()
    rv = np.random.rand(base_cell_data.size)
    s_idx = rv < salt_prob
    p_idx = (rv >= salt_prob) & (rv < salt_prob + pepper_prob)
    ni_cells[s_idx] = 1.0
    ni_cells[p_idx] = 0.0

    # Store first S&P noisy cells for visualization
    if i == 0:
        mask_cells = s_idx | p_idx
        noise_cells_viz = cell_mesh.extract_cells(np.where(mask_cells)[0])
        noise_cells_viz.cell_data["intensity"] = ni_cells[mask_cells]

    # Save cell-based stack for visualization
    temp_cell = cell_mesh.copy()
    temp_cell.cell_data["intensity"] = ni_cells
    sp_stack_cells.append(temp_cell.cell_data["intensity"])
    f_sp = temp_cell.cell_data_to_point_data().point_data["intensity"]
    sp_stack_cells_as_points.append(f_sp)

    # S&P noise (point-based)
    ni_points = ref_intensities.copy()
    rv_points = np.random.rand(ni_points.size)
    s_idx_p = rv_points < salt_prob
    p_idx_p = (rv_points >= salt_prob) & (rv_points < salt_prob + pepper_prob)
    ni_points[s_idx_p] = 1.0
    ni_points[p_idx_p] = 0.0
    sp_stack_points.append(ni_points)

    # Gaussian Noise (point-based)
    f_g_points = np.clip(ref_intensities + np.random.normal(0, 1e-1, ref_intensities.shape), 0, 1)
    gauss_stack_points.append(f_g_points)

# Convert stacks to numpy arrays
sp_stack_points = np.stack(sp_stack_points)
gauss_stack_points = np.stack(gauss_stack_points)
sp_stack_cells = np.stack(sp_stack_cells)


# 3. 3D VISUALIZATIONS
def plot_3d(volume_mesh, title, extra_mesh=None):
    pl = pv.Plotter(title=title)
    pl.add_volume(volume_mesh, scalars="intensity", cmap=cmap, clim=[0, 1], opacity=opacity_transfer)
    if extra_mesh:
        pl.add_mesh(extra_mesh, scalars="intensity", cmap=cmap, clim=[0, 1], opacity=1.0)
    pl.view_zx()
    pl.show_grid()
    pl.show()

def plot_slice(volume_mesh, title, extra_mesh=None):
    # 2D Slice View
    pl_slice = pv.Plotter(title=f"{title} (Mid-Z Slice)")
    slc = volume_mesh.slice(normal='z', origin=(0, 0, z_mid))
    pl_slice.add_mesh(slc, scalars="intensity", cmap=cmap, clim=[0, 1])
    if extra_mesh:
        # Slice the extra S&P cells too
        slc_extra = extra_mesh.slice(normal='z', origin=(0, 0, z_mid))
        pl_slice.add_mesh(slc_extra, scalars="intensity", cmap=cmap, clim=[0, 1])
    pl_slice.view_yx()  # View flat on the slice
    pl_slice.show()

# A new mesh is created for the Gaussian noisy mesh to avoid attribute error
gauss_viz_mesh = mesh.copy()
gauss_viz_mesh.point_data["intensity"] = gauss_stack_points[0]


def plot_3d_comparison():
    pl = pv.Plotter(shape=(1, 3), title="3D Comparison: Original vs Noise")

    # Original
    pl.subplot(0, 0)
    pl.add_volume(mesh, scalars="intensity", cmap=cmap, clim=[0,1], opacity=opacity_transfer)
    pl.add_text("Original", font_size=12)

    # Salt & Pepper
    pl.subplot(0, 1)
    pl.add_volume(mesh, scalars="intensity", cmap=cmap, clim=[0,1], opacity=opacity_transfer)
    pl.add_mesh(noise_cells_viz, scalars="intensity", cmap=cmap, clim=[0,1], opacity=1.0)
    pl.add_text("Salt & Pepper Noise", font_size=12)

    # Gaussian
    pl.subplot(0, 2)
    gauss_viz_mesh = mesh.copy()
    gauss_viz_mesh.point_data["intensity"] = gauss_stack_points[0]
    pl.add_volume(gauss_viz_mesh, scalars="intensity", cmap=cmap, clim=[0,1], opacity=opacity_transfer)
    pl.add_text("Gaussian Noise", font_size=12)

    pl.link_views()
    pl.view_zx()
    pl.camera.zoom(1.5)
    pl.show(interactive_update=False) # # use True when saving screenshot

def plot_slice_comparison():
    pl = pv.Plotter(shape=(1, 3), title="2D Mid-Z Slice Comparison")

    # Original slice
    pl.subplot(0, 0)
    slc_orig = mesh.slice(normal='z', origin=(0,0,z_mid))
    pl.add_mesh(slc_orig, scalars="intensity", cmap=cmap, clim=[0,1])
    pl.add_text("Original", font_size=12)
    pl.view_yx()

    # S&P slice
    pl.subplot(0, 1)
    slc_sp = mesh.slice(normal='z', origin=(0,0,z_mid))
    pl.add_mesh(slc_sp, scalars="intensity", cmap=cmap, clim=[0,1])
    slc_noise = noise_cells_viz.slice(normal='z', origin=(0,0,z_mid))
    pl.add_mesh(slc_noise, scalars="intensity", cmap=cmap, clim=[0,1])
    pl.add_text("Salt & Pepper Noise", font_size=12)
    pl.view_yx()

    # Gaussian slice
    pl.subplot(0, 2)
    slc_gauss = gauss_viz_mesh.slice(normal='z', origin=(0,0,z_mid))
    pl.add_mesh(slc_gauss, scalars="intensity", cmap=cmap, clim=[0,1])
    pl.add_text("Gaussian Noise", font_size=12)
    pl.view_yx()

    pl.link_views()
    pl.camera.zoom(1.35)
    pl.show(interactive_update=False) # # use True when saving screenshot
    #pl.screenshot("Task_6_slice_comparison.png")
    #pl.close()
    #img = Image.open("Task_6_slice_comparison.png")
    #img = img.convert("RGB")
    #img.save("Task_6_slice_comparison.pdf")


def plot_3d_recon_comparison():
    pl = pv.Plotter(shape=(2, 2), title="3D Reconstructions Comparison")

    for i, (name, data) in enumerate(recon_results.items()):
        row, col = divmod(i, 2)
        pl.subplot(row, col)
        temp_mesh = mesh.copy()
        temp_mesh.point_data["intensity"] = data
        pl.add_volume(temp_mesh, scalars="intensity", cmap=cmap, clim=[0, 1], opacity=opacity_transfer, show_scalar_bar=False)
        pl.add_text(name, font_size=12)
        pl.add_scalar_bar(title="Intensity", vertical=True, n_labels=5, height=0.7, position_x=0.83, position_y=0.2)

    pl.link_views()
    pl.view_zx()
    pl.camera.zoom(1.5)
    pl.show(interactive_update=False) # use True when saving screenshot
    #pl.screenshot("Task_6_3d_recon_comparison.png")
    #pl.close()
    #img = Image.open("Task_6_3d_recon_comparison.png")
    #img = img.convert("RGB")
    #img.save("Task_6_3d_recon_comparison.pdf")

def plot_slice_recon_comparison():
    pl = pv.Plotter(shape=(2, 2), title="2D Mid-Z Slice Reconstructions")

    for i, (name, data) in enumerate(recon_results.items()):
        row, col = divmod(i, 2)
        pl.subplot(row, col)
        temp_mesh = mesh.copy()
        temp_mesh.point_data["intensity"] = data
        slc = temp_mesh.slice(normal='z', origin=(0, 0, z_mid))
        pl.add_mesh(slc, scalars="intensity", cmap=cmap, clim=[0, 1], show_scalar_bar=False)
        pl.add_text(name, font_size=12)
        pl.add_scalar_bar(title="Intensity", vertical=True, n_labels=5, height=0.7, position_x=0.86, position_y=0.2)
        pl.view_yx()

    pl.link_views()
    pl.camera.zoom(1.2)
    pl.show(interactive_update=False) # use True when saving screenshot
    #pl.screenshot("Task_6_slice_recon_comparison.png")
    #pl.close()
    #img = Image.open("Task_6_slice_recon_comparison.png")
    #img = img.convert("RGB")
    #img.save("Task_6_slice_recon_comparison.pdf")

plot_slice(mesh, "Original")
plot_3d(mesh, "Salt & Pepper Noise", extra_mesh=noise_cells_viz)
plot_slice(mesh, "Salt & Pepper Noise", extra_mesh=noise_cells_viz)
plot_3d(gauss_viz_mesh, "Gaussian Noise")
plot_slice(gauss_viz_mesh, "Gaussian Noise")

# 4. CROSS-RECONSTRUCTION VISUALS
recon_results = {
    "S&P Median": np.nanmedian(sp_stack_points, axis=0), # axis=0 is crucial for the filter to work per voxel
    "S&P Mean": np.nanmean(sp_stack_points, axis=0),
    "Gauss Mean": np.nanmean(gauss_stack_points, axis=0),
    "Gauss Median": np.nanmedian(gauss_stack_points, axis=0)
}

plot_3d_comparison()
plot_slice_comparison()
plot_3d_recon_comparison()
plot_slice_recon_comparison()

# 5. HISTOGRAMS
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

def plot_hist(ax, original, noisy, recon, title):
    # Only plot foreground voxels
    num_bins = 256
    ax.hist(noisy, bins=num_bins, log=True, color='red', alpha=0.5, label='Noisy')
    ax.hist(recon, bins=num_bins, log=True, color='green', alpha=0.5, label='Reconstructed')
    ax.hist(original, bins=num_bins, log=True, color='blue', alpha=0.5, label='Original')
    ax.set_title(title)
    ax.legend()

# Plots histograms for the whole volume
plot_hist(axes[0, 0], ref_intensities, sp_stack_points[0], recon_results["S&P Median"], "S&P: Median Filter")
plot_hist(axes[0, 1], ref_intensities, sp_stack_points[0], recon_results["S&P Mean"], "S&P: Mean Filter")
plot_hist(axes[1, 0], ref_intensities, gauss_stack_points[0], recon_results["Gauss Mean"], "Gauss: Mean Filter")
plot_hist(axes[1, 1], ref_intensities, gauss_stack_points[0], recon_results["Gauss Median"],  "Gauss: Median Filter")
plt.tight_layout()
plt.show()

# 6. PSNR/SSIM for the 27 average reconstruction
ref_v = ref_intensities[h_mask_3d]
ref_v_gauss = ref_intensities[h_mask_3d]
s_med, s_mea = np.nanmedian(sp_stack_points[:n_reconstructions], axis=0)[h_mask_3d], np.nanmean(sp_stack_points[:n_reconstructions], axis=0)[h_mask_3d]
g_mea, g_med = np.nanmean(gauss_stack_points[:n_reconstructions], axis=0)[h_mask_3d], np.nanmedian(gauss_stack_points[:n_reconstructions], axis=0)[h_mask_3d]

s_med_ssim, s_mea_ssim = np.nanmedian(sp_stack_cells_as_points[:n_reconstructions], axis=0)[h_mask_3d], np.nanmean(sp_stack_cells_as_points[:n_reconstructions], axis=0)[h_mask_3d]

# Column headers
headers = ["", "PSNR", "SSIM"]

# Rows
rows = [
    ["S&P Mean",   f"{psnr(ref_v, s_mea, data_range=1.0):.2g}", f"{ssim(ref_v, s_mea_ssim, data_range=1.0):.2g}"],
    ["S&P Median", f"{psnr(ref_v, s_med, data_range=1.0):.2g}", f"{ssim(ref_v, s_med_ssim, data_range=1.0):.2g}"],
    ["Gauss Mean", f"{psnr(ref_v_gauss, g_mea, data_range=1.0):.2g}", f"{ssim(ref_v_gauss, g_mea, data_range=1.0):.2g}"],
    ["Gauss Median", f"{psnr(ref_v_gauss, g_med, data_range=1.0):.2g}", f"{ssim(ref_v_gauss, g_med, data_range=1.0):.2g}"]
]

# Print header
print(f"{headers[0]:<15}{headers[1]:>10}{headers[2]:>10}")

# Print rows
for r in rows:
    print(f"{r[0]:<15}{r[1]:>10}{r[2]:>10}")

# PSNR/SSIM Analysis Loop
n_range = range(2, n_reconstructions + 1)
metrics = {k: [] for k in
           ['sp_med_p', 'sp_mean_p', 'gauss_mean_p', 'gauss_med_p', 'sp_med_s', 'sp_mean_s', 'gauss_mean_s',
            'gauss_med_s']}

def safe_psnr(ref, target, data_range=1.0, eps=1e-10):
    # Computes PSNR from the Mean Square Error (MSE)
    # The PSNR is limited at 100dB for clearer visuals
    mse = np.mean((ref - target) ** 2)
    if mse < eps:
        return 100  # perfect reconstruction
    return 10 * np.log10((data_range ** 2) / mse)

for n in n_range:

    s_med, s_mea = np.nanmedian(sp_stack_points[:n], axis=0)[h_mask_3d], np.nanmean(sp_stack_points[:n], axis=0)[h_mask_3d]
    g_mea, g_med = np.nanmean(gauss_stack_points[:n], axis=0)[h_mask_3d], np.nanmedian(gauss_stack_points[:n], axis=0)[h_mask_3d]
    s_med_ssim, s_mea_ssim = np.nanmedian(sp_stack_cells_as_points[:n], axis=0)[h_mask_3d], np.nanmean(sp_stack_cells_as_points[:n], axis=0)[h_mask_3d]

    metrics['sp_med_p'].append(safe_psnr(ref_v, s_med, data_range=1.0))
    metrics['sp_mean_p'].append(psnr(ref_v, s_mea, data_range=1.0))
    metrics['gauss_mean_p'].append(psnr(ref_v_gauss, g_mea, data_range=1.0))
    metrics['gauss_med_p'].append(psnr(ref_v_gauss, g_med, data_range=1.0))

    metrics['sp_med_s'].append(ssim(ref_v, s_med_ssim, data_range=1.0))
    metrics['sp_mean_s'].append(ssim(ref_v, s_mea_ssim, data_range=1.0))
    metrics['gauss_mean_s'].append(ssim(ref_v_gauss, g_mea, data_range=1.0))
    metrics['gauss_med_s'].append(ssim(ref_v_gauss, g_med, data_range=1.0))

# Plot PSNR
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(n_range, metrics['sp_med_p'], 'g-', label="S&P Median")
plt.plot(n_range, metrics['sp_mean_p'], 'g--', label="S&P Mean")
plt.plot(n_range, metrics['gauss_mean_p'], 'b-', label="Gauss Mean")
plt.plot(n_range, metrics['gauss_med_p'], 'b--', label="Gauss Median")
plt.title("PSNR Comparison")
plt.xlabel("N Samples")
plt.ylabel("PSNR [dB]")
plt.legend()

# Plot SSIM
plt.subplot(1, 2, 2)
plt.plot(n_range, metrics['sp_med_s'], 'g-', label="S&P Median")
plt.plot(n_range, metrics['sp_mean_s'], 'g--', label="S&P Mean")
plt.plot(n_range, metrics['gauss_mean_s'], 'b-', label="Gauss Mean")
plt.plot(n_range, metrics['gauss_med_s'], 'b--', label="Gauss Median")
plt.title("SSIM Comparison")
plt.xlabel("N Samples")
plt.ylabel("SSIM")
plt.legend()
plt.tight_layout()
plt.show()