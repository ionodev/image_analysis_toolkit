import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float64, img_as_ubyte, io
from scipy.ndimage import uniform_filter, gaussian_filter1d

def compute_fft(img):
    # Apply 2D Fourier Transform
    F = np.fft.fft2(img)  # 2D FFT
    F_shifted = np.fft.fftshift(F)  # Shift zero frequency to center
    return F_shifted

def magnitude_spectrum(F):
    # Compute magnitude spectrum (log scale for visualization)
    magnitude_spectrum = np.log1p(np.abs(F))
    return magnitude_spectrum

def phase_spectrum(F):
    # Return phase spectrum
    return np.angle(F)

def local_var(img):
    # Local variance map
    local_mean = uniform_filter(img, size=3)
    local_sq_mean = uniform_filter(img ** 2, size=3)
    local_var = local_sq_mean - local_mean ** 2
    return local_var

def CV(img):
    # Coefficient of variation: sigma/mu
    local_mean = uniform_filter(img, size=3)
    local_sq_mean = uniform_filter(img ** 2, size=3)
    local_var = local_sq_mean - local_mean ** 2
    return np.sqrt(local_var) / local_mean

# Load data
img_A = img_as_float64(io.imread('data/Task1_ImageA.jpg', as_gray=True))
img_B = img_as_float64(io.imread('data/Task1_ImageB.jpg', as_gray=True))
img_C = img_as_float64(io.imread('data/Task1_ImageC.jpg', as_gray=True))
img_D = img_as_float64(io.imread('data/Task1_ImageD.jpg', as_gray=True))

img_A_mag = magnitude_spectrum(compute_fft(img_A))
img_B_mag = magnitude_spectrum(compute_fft(img_B))
img_C_mag = magnitude_spectrum(compute_fft(img_C))
img_D_mag = magnitude_spectrum(compute_fft(img_D))

img_A_phase = phase_spectrum(compute_fft(img_A))
img_B_phase = phase_spectrum(compute_fft(img_B))
img_C_phase = phase_spectrum(compute_fft(img_C))
img_D_phase = phase_spectrum(compute_fft(img_D))


# Plot degraded images as is
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

ax[0, 0].imshow(img_A, cmap='gray')
ax[0, 0].axis('off')
ax[0, 0].set_title('A')
ax[0, 1].imshow(img_B, cmap='gray')
ax[0, 1].axis('off')
ax[0, 1].set_title('B')
ax[1, 0].imshow(img_C, cmap='gray')
ax[1, 0].axis('off')
ax[1, 0].set_title('C')
ax[1, 1].imshow(img_D, cmap='gray')
ax[1, 1].axis('off')
ax[1, 1].set_title('D')

plt.tight_layout()
#plt.savefig("images_comparison.pdf", bbox_inches='tight')
plt.show()

# Plot magnitude spectrum
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

im0 = ax[0, 0].imshow(img_A_mag, cmap='magma', extent=[-0.5, 0.5, -0.5, 0.5])
ax[0, 0].set_title('A')
cbar0 = fig.colorbar(im0, ax=ax[0, 0])
cbar0.set_label("Log Magnitude")
ax[0, 0].set_xlabel(f"$f_x$ [cycles/pixel]")
ax[0, 0].set_ylabel(f"$f_y$ [cycles/pixel]")

im1 = ax[0, 1].imshow(img_B_mag, cmap='magma', extent=[-0.5, 0.5, -0.5, 0.5])
ax[0, 1].set_title('B')
cbar1 = fig.colorbar(im1, ax=ax[0, 1])
cbar1.set_label("Log Magnitude")
ax[1, 0].set_xlabel(f"$f_x$ [cycles/pixel]")
ax[1, 0].set_ylabel(f"$f_y$ [cycles/pixel]")

im2 = ax[1, 0].imshow(img_C_mag, cmap='magma', extent=[-0.5, 0.5, -0.5, 0.5])
ax[1, 0].set_title('C')
cbar2 = fig.colorbar(im2, ax=ax[1, 0])
cbar2.set_label("Log Magnitude")
ax[1, 0].set_xlabel(f"$f_x$ [cycles/pixel]")
ax[1, 0].set_ylabel(f"$f_y$ [cycles/pixel]")

im3 = ax[1, 1].imshow(img_D_mag, cmap='magma', extent=[-0.5, 0.5, -0.5, 0.5])
ax[1, 1].set_title('D')
cbar3 = fig.colorbar(im3, ax=ax[1, 1])
cbar3.set_label("Log Magnitude")
ax[1, 1].set_xlabel(f"$f_x$ [cycles/pixel]")
ax[1, 1].set_ylabel(f"$f_y$ [cycles/pixel]")

plt.tight_layout()
#plt.savefig("Task_1_images/magnitude_spectrum.pdf", bbox_inches='tight')
plt.show()

# Plot phase spectrum
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

im0 = ax[0, 0].imshow(img_A_phase, cmap='hsv', extent=[-0.5, 0.5, -0.5, 0.5])
ax[0, 0].set_title('A')
cbar0 = fig.colorbar(im0, ax=ax[0, 0])
cbar0.set_label("Phase [radians]")
ax[0, 0].set_xlabel(f"$f_x$ [cycles/pixel]")
ax[0, 0].set_ylabel(f"$f_y$ [cycles/pixel]")

im1 = ax[0, 1].imshow(img_B_phase, cmap='hsv', extent=[-0.5, 0.5, -0.5, 0.5])
ax[0, 1].set_title('B')
cbar1 = fig.colorbar(im1, ax=ax[0, 1])
cbar1.set_label("Phase [radians]")
ax[0, 1].set_xlabel(f"$f_x$ [cycles/pixel]")
ax[0, 1].set_ylabel(f"$f_y$ [cycles/pixel]")

im2 = ax[1, 0].imshow(img_C_phase, cmap='hsv', extent=[-0.5, 0.5, -0.5, 0.5])
ax[1, 0].set_title('C')
cbar2 = fig.colorbar(im2, ax=ax[1, 0])
cbar2.set_label("Phase [radians]")
ax[1, 0].set_xlabel(f"$f_x$ [cycles/pixel]")
ax[1, 0].set_ylabel(f"$f_y$ [cycles/pixel]")

im3 = ax[1, 1].imshow(img_D_phase, cmap='hsv', extent=[-0.5, 0.5, -0.5, 0.5])
ax[1, 1].set_title('D')
cbar3 = fig.colorbar(im3, ax=ax[1, 1])
cbar3.set_label("Phase [radians]")
ax[1, 1].set_xlabel(f"$f_x$ [cycles/pixel]")
ax[1, 1].set_ylabel(f"$f_y$ [cycles/pixel]")

plt.tight_layout()
#plt.savefig("Task_1_images/phase_spectrum.pdf", bbox_inches='tight')
plt.show()

# Plot local variance maps
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

im0 = ax[0, 0].imshow(local_var(img_A), cmap='magma')
ax[0, 0].set_title('A')
ax[0, 0].axis('off')
cbar0 = fig.colorbar(im0, ax=ax[0, 0])
cbar0.set_label("Local Variance")

im1 = ax[0, 1].imshow(local_var(img_B), cmap='magma')
ax[0, 1].set_title('B')
ax[0, 1].axis('off')
cbar1 = fig.colorbar(im1, ax=ax[0, 1])
cbar1.set_label("Local Variance")

im2 = ax[1, 0].imshow(local_var(img_C), cmap='magma')
ax[1, 0].set_title('C')
ax[1, 0].axis('off')
cbar2 = fig.colorbar(im2, ax=ax[1, 0])
cbar2.set_label("Local Variance")

im3 = ax[1, 1].imshow(local_var(img_D), cmap='magma')
ax[1, 1].set_title('D')
ax[1, 1].axis('off')
cbar3 = fig.colorbar(im3, ax=ax[1, 1])
cbar3.set_label("Local Variance")

plt.tight_layout()
#plt.savefig("Task_1_images/local_variance.pdf", bbox_inches='tight')
plt.show()


# Plot CV maps
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

im0 = ax[0, 0].imshow(CV(img_A), cmap='magma')
ax[0, 0].set_title('A')
ax[0, 0].axis('off')
cbar0 = fig.colorbar(im0, ax=ax[0, 0])
cbar0.set_label("Coefficient of Variation")

im1 = ax[0, 1].imshow(CV(img_B), cmap='magma')
ax[0, 1].set_title('B')
ax[0, 1].axis('off')
cbar1 = fig.colorbar(im1, ax=ax[0, 1])
cbar1.set_label("Coefficient of Variation")

im2 = ax[1, 0].imshow(CV(img_C), cmap='magma')
ax[1, 0].set_title('C')
ax[1, 0].axis('off')
cbar2 = fig.colorbar(im2, ax=ax[1, 0])
cbar2.set_label("Coefficient of Variation")

im3 = ax[1, 1].imshow(CV(img_D), cmap='magma')
ax[1, 1].set_title('D')
ax[1, 1].axis('off')
cbar3 = fig.colorbar(im3, ax=ax[1, 1])
cbar3.set_label("Coefficient of Variation")

plt.tight_layout()
#plt.savefig("Task_1_images/CV.pdf", bbox_inches='tight')
plt.show()

# Plot histograms
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

ax[0,0].hist(img_A.ravel(), bins=256, density=True, range=(0,1))
ax[0, 0].set_title('A')
ax[0, 0].set_ylim(top=4.5)
ax[0, 0].set_ylabel('Probability density')
ax[0, 0].set_xlabel('Intensity')
ax[0,1].hist(img_B.ravel(), bins=256, density=True, range=(0,1))
ax[0, 1].set_title('B')
ax[0, 1].set_ylim(top=4.5)
ax[0, 1].set_ylabel('Probability density')
ax[0, 1].set_xlabel('Intensity')
ax[1,0].hist(img_C.ravel(), bins=256, density=True, range=(0,1))
ax[1, 0].set_title('C')
ax[1, 0].set_ylim(top=4.5)
ax[1, 0].set_ylabel('Probability density')
ax[1, 0].set_xlabel('Intensity')
ax[1,1].hist(img_D.ravel(), bins=256, density=True, range=(0,1))
ax[1, 1].set_title('D')
ax[1, 1].set_ylim(top=4.5)
ax[1, 1].set_ylabel('Probability density')
ax[1, 1].set_xlabel('Intensity')
#plt.savefig("Task_1_images/histograms.png", bbox_inches='tight')
plt.tight_layout()

plt.show()

# Log transformed version of image D, gaussian filtered and normalized for clearer visualization
img_D_log = (gaussian_filter1d(np.log1p(img_D), sigma=0.4) / np.log1p(1))

# Histogram for Image A
hist_A, bins_A = np.histogram(img_A.ravel(), bins=256, range=(0,1), density=True)

# Histogram for Image D
hist_D, bins_D = np.histogram(img_D.ravel(), bins=256, range=(0,1), density=True)

# Histogram for Image log D
hist_D_log, bins_D_log = np.histogram(img_D_log.ravel(), bins=256, range=(0,1), density=True)

# Plot histogram comparison A vs. D
plt.figure(figsize=(8,5))
plt.plot(bins_A[:-1], hist_A, label='Image A', color='blue')
plt.plot(bins_D[:-1], hist_D, label='Image D', color='red')
plt.xlabel('Intensity')
plt.ylabel('Probability density')
plt.legend()
plt.grid(True)
#plt.savefig("Task_1_images/histogram_close_up_A_vs_D.pdf", bbox_inches='tight')
plt.show()

# Plot histogram comparison A, D, log D, shifted for aligned peaks
plt.figure(figsize=(8,5))
plt.plot(bins_A[:-1], hist_A, label='Image A', color='blue')
plt.plot(bins_D[:-1]+0.02, hist_D, label='Image D', color='red')
plt.plot(bins_D_log[:-1] - 0.07, hist_D_log, label='Image D log', color='green')
plt.xlabel('Intensity')
plt.ylabel('Probability density')
plt.legend()
plt.grid(True)
#plt.savefig("Task_1_images/histograms_close_up_aligned.pdf", bbox_inches='tight')
plt.show()
