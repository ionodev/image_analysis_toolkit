import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float64
from scipy.ndimage import laplace, map_coordinates, gaussian_filter
from skimage.feature import structure_tensor
from skimage.transform import resize

def compute_fft(img):
    # Return centered 2D FFT
    return np.fft.fftshift(np.fft.fft2(img))

def magnitude_spectrum(F, log=True):
    # Return magnitude spectrum
    mag = np.abs(F)
    return np.log1p(mag) if log else mag

def phase_spectrum(F):
    # Return phase spectrum
    return np.angle(F)

def phase_first_derivative(phase):
    # Return gradient magnitude of phase
    gy, gx = np.gradient(phase)
    return np.sqrt(gx**2 + gy**2)

def phase_second_derivative(phase):
    # Return Laplacian (2nd derivative) of phase
    return np.abs(laplace(phase))

def plot_spectra(images, type=None, titles=None, cmap='magma',
                 vmin=None, vmax=None, cols=2):

    n = len(images)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    axes = np.ravel(axes)

    for i, img in enumerate(images):

        im = axes[i].imshow(
            img,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[-0.5, 0.5, -0.5, 0.5]
        )

        axes[i].set_title(titles[i] if titles else f"Image {i}")
        axes[i].set_xlabel(r"$f_x$ [cycles/pixel]")
        axes[i].set_ylabel(r"$f_y$ [cycles/pixel]")
        cbar = plt.colorbar(im, ax=axes[i])

        if type == 'phase':
            cbar.set_label("Phase [radians]")
        elif type == 'phase-grad':
            cbar.set_label("Gradient of phase [radians/(cycles/pixel)]")

        elif type == 'phase-curv':
            cbar.set_label(f"Laplacian of phase [radians/(cycles/pixel)$^2$]")
        elif type == 'phase-orientation':
            cbar.set_label("Local Phase Orientation [deg °]")
        else:
            cbar.set_label("Log Magnitude")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def plot_images(images, titles=None, cols=2):
    # Used for plotting the original brochure images

    n = len(images)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    axes = np.ravel(axes)
    im = None

    for i, img in enumerate(images):
        im = axes[i].imshow(img, cmap='gray')
        axes[i].set_title(titles[i] if titles else f"Image {i}")
        axes[i].set_xlabel("x [pixels]")
        axes[i].set_ylabel("y [pixels]")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def plot_slices(data, img_letter=None):
    # Used for plotting the angular slice analysis

    def get_angular_slice(mag_spectrum, angle_deg, num_points=500, delta=5):
        """
        Helper function:
        Extracts the angular slice around angle_deg ± delta and chooses the strongest one.
        """

        h, w = mag_spectrum.shape
        cy, cx = h // 2, w // 2
        max_r = min(h, w) // 2

        r = np.linspace(-max_r, max_r, num_points)
        freqs = np.linspace(-0.5, 0.5, num_points)

        best_slice = None
        best_sum = -np.inf

        for ang in range(angle_deg - delta, angle_deg + delta + 1):

            x_coords = cx + r * np.cos(np.radians(ang))
            y_coords = cy - r * np.sin(np.radians(ang))

            slice_vals = map_coordinates(mag_spectrum, [y_coords, x_coords], order=1)

            total_val = slice_vals.sum()

            if total_val > best_sum:
                best_sum = total_val
                best_slice = slice_vals

        return freqs, best_slice


    tasks = {
        'A': [0, 45, 90],
        'B': [30, 75, 120],
        'C': [60, 105, 150],
        'D': [75, 120, 165]
    }
    slice_store = {}

    for i, (key, angles) in enumerate(tasks.items()):

        mag = data[i]

        for ang in angles:
            f, s = get_angular_slice(mag, ang)
            slice_store[(key, ang)] = (f, s)

    angle_groups = [
        [0, 30, 60, 75],
        [45, 75, 105, 120],
        [90, 120, 150, 165]
    ]

    image_keys = ['A', 'B', 'C', 'D']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    titles_cmp = [
        "Parallel Line Angles Comparison",
        "Diagonal Line Angles Comparison",
        "Perpendicular Line Angles Comparison"
    ]

    for g, ax in enumerate(axes):

        for img_key, ang in zip(image_keys, angle_groups[g]):

            f, s = slice_store[(img_key, ang)]

            if img_letter:
                if img_key == img_letter:
                    ax.plot(f, s, label=f"Image {img_key} ({ang}°)")
            else:
                ax.plot(f, s, label=f"Image {img_key} ({ang}°)")


        ax.set_title(titles_cmp[g])
        ax.set_xlabel("Normalized Frequency [cycles/pixel]")
        ax.set_ylabel("Log Magnitude")
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([2, 11])
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

def reconstruct_from_mag_phase(mag, phase):
    """
    Reconstruct spatial image from magnitude and phase.
    """
    F = mag * np.exp(1j * phase) # Combine magnitude and phase
    F_ishift = np.fft.ifftshift(F) # Undo shift
    img_recon = np.fft.irfft2(F_ishift) # Inverse FFT
    return img_recon


def plot_phase_orientation_spread(phases, titles=None, bins=256):
    """
    Compute and plot the orientation spread of the phase gradient
    """

    n = len(phases)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

    if n == 1:
        axes = [axes]

    spreads = []

    for i, phase in enumerate(phases):

        # Phase gradients
        gy, gx = np.gradient(phase)

        # Gradient orientation
        theta = np.arctan2(gy, gx)

        # Flatten
        theta_flat = theta.ravel()

        # Angular spread (variance)
        #spread = np.var(theta_flat)
        R = np.sqrt(np.mean(np.cos(theta_flat)) ** 2 + np.mean(np.sin(theta_flat)) ** 2)
        spread = 1 - R
        spreads.append(spread)

        # Histogram
        axes[i].hist(theta_flat, bins=bins, range=(-np.pi, np.pi), density=True)

        axes[i].set_title(
            f"{titles[i] if titles else f'Image {i}'}\nSpread={spread:.3f}"
        )
        axes[i].set_xlabel(r"$\theta = \arctan(\partial\phi_y / \partial\phi_x)$")
        axes[i].set_ylabel("Probability Density")

    plt.tight_layout()
    plt.show()

    return spreads

def phase_coherence_plot(images, titles=None):
    n_angle_bins = 180  # number of angular bins
    n_images = len(images)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()  # flatten for easy indexing

    for idx, image in enumerate(images):
        if idx >= 4:
            print("Warning: only the first 4 images will be plotted in 2x2 grid.")
            break

        # FFT and phase
        F = np.fft.fftshift(np.fft.fft2(image))
        phase = np.angle(F)
        magnitude = np.abs(F)

        h, w = phase.shape

        # Frequency coordinate grid
        y, x = np.indices((h, w))
        cy, cx = h // 2, w // 2
        x = x - cx
        y = y - cy

        freq_angle = np.arctan2(y, x)  # angle of each frequency component [-π, π]

        # Bin by frequency angle
        bins = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
        digitized = np.digitize(freq_angle.ravel(), bins) - 1
        digitized = np.clip(digitized, 0, n_angle_bins - 1)

        # Compute phase coherence per angle bin
        phase_vals = phase.ravel()
        coherence = np.zeros(n_angle_bins)

        for i in range(n_angle_bins):
            bin_mask = digitized == i
            if np.any(bin_mask):
                phases_bin = phase_vals[bin_mask]
                coherence[i] = np.abs(np.sum(np.exp(1j * phases_bin))) / len(phases_bin)

        # Angle centers for plotting
        angle_centers = (bins[:-1] + bins[1:]) / 2

        # Plot on the subplot
        ax = axes[idx]
        ax.plot(np.degrees(angle_centers), coherence, lw=2)
        ax.set_xlabel("Frequency angle (°)")
        ax.set_ylabel("Phase coherence")
        ax.set_title(titles[idx] if titles else f"Image {i}")
        ax.grid(True)

    # Hide unused subplots if less than 4 images
    for j in range(idx + 1, 4):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_phase_orientation_histograms(images, titles, method="gradient", bins=360):

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for i, image in enumerate(images):

        # FFT phase
        F = np.fft.fftshift(np.fft.fft2(image))
        phase = np.angle(F)

        if method == "gradient":
            py, px = np.gradient(phase)
            theta = np.arctan2(py, px)
            xlim = (-180, 180)

        elif method == "structure_tensor":
            Axx, Axy, Ayy = structure_tensor(phase)
            theta = 0.5 * np.arctan2(2*Axy, Axx - Ayy)
            xlim = (-90, 90)
        elif method == 'phase':
            theta = phase
            xlim = (-180, 180)
        else:
            raise ValueError("method must be 'gradient' or 'structure_tensor'")

        ax = axes[i]

        ax.hist(np.degrees(theta).ravel(), bins=bins, density=True)
        ax.set_title(titles[i])
        if method != 'phase':
            ax.set_xlabel("Phase Orientation [deg °]")
        else:
            ax.set_xlabel("Phase [deg °]")
        ax.set_ylabel("Probability Density")
        ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()


# Load images
titles = ["A: 0°", "B: 30°", "C: 60°", "D: 75°"]

img_A = img_as_float64(io.imread('data/Task2_ImageA.png', as_gray=True))
img_B = img_as_float64(io.imread('data/Task2_ImageB.png', as_gray=True))
img_C = img_as_float64(io.imread('data/Task2_ImageC.png', as_gray=True))
img_D = img_as_float64(io.imread('data/Task2_ImageD.png', as_gray=True))
images = [img_A, img_B, img_C, img_D]

# FFT analysis
ffts = [compute_fft(img) for img in images]
magnitudes = [magnitude_spectrum(F) for F in ffts]
phases = [phase_spectrum(F) for F in ffts]
phase_grad = [phase_first_derivative(p) for p in phases]
phase_curv = [phase_second_derivative(p) for p in phases]

# Phase orientation
def phase_orientation(phase, degrees=False):
    # local phase orientation angle in radians
    Axx, Axy, Ayy = structure_tensor(phase)
    theta = 0.5 * np.arctan2(2 * Axy, Axx - Ayy)
    if degrees:
        theta = np.degrees(theta)
    return theta

phase_orientation_angles = [phase_orientation(p, degrees=True) for p in phases]

# Plot brochure images
plot_images(images, titles=titles)

# Plot spectra
plot_spectra(magnitudes, titles=titles, cmap="jet", vmin=3, vmax=7)
plot_spectra(phases, type='phase', titles=titles, cmap='hsv')
plot_spectra(phase_grad, type='phase-grad', titles=titles, cmap="seismic", vmax=4)
plot_spectra(phase_curv, type='phase-curv', titles=titles, cmap="seismic")
plot_spectra(phase_orientation_angles, type='phase-orientation', titles=titles, cmap='hsv')

# Plot slices of the magnitue spectrum
plot_slices(magnitudes)


# Plot histogram of the phase and phase orientation histograms for the gradient/structure tensor of the phase
plot_phase_orientation_histograms(images, titles=titles, method='phase')
plot_phase_orientation_histograms(images, titles=titles, method='gradient')
plot_phase_orientation_histograms(images, titles=titles, method='structure_tensor')

# Plot the phase coherence
phase_coherence_plot(images, titles=titles)


# Resize C to match A
img_C_resized = resize(img_C, img_A.shape, anti_aliasing=True)

# FFT
F_A = compute_fft(img_A)
F_C = compute_fft(img_C_resized)

# Magnitude and phase
mag_A = np.abs(F_A)
phase_A = np.angle(F_A)

mag_C = np.abs(F_C)
phase_C = np.angle(F_C)

# Reconstruction
recon_A_mag_C_phase = reconstruct_from_mag_phase(mag_A, phase_C)
recon_C_mag_A_phase = reconstruct_from_mag_phase(mag_C, phase_A)

fig, axes = plt.subplots(1, 2, figsize=(12, 7))

im0 = axes[0].imshow(
    recon_A_mag_C_phase,
    cmap='gray',
    extent=[0, 1, 0, 1]
)
axes[0].set_title("A magnitude + C phase")
axes[0].set_xlabel(f"x (normalized)")
axes[0].set_ylabel(f"y (normalized)")

im1 = axes[1].imshow(
    recon_C_mag_A_phase,
    cmap='gray',
    extent=[0, 1, 0, 1]
)
axes[1].set_title("C magnitude + A phase")
axes[1].set_xlabel(f"x (normalized)")
axes[1].set_ylabel(f"y (normalized)")

fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()