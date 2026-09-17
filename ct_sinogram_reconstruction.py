import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def inverse_radon(sinogram, theta, image_size=None, filter_type='ramp', cutoff=0.3):

    """
    Inverse Radon Transform for non-square sinograms.
    
    Args:
        sinogram (np.ndarray): 2D array of shape (num_angles, num_detectors)
        theta (np.ndarray): 1D array of projection angles in degrees
        image_size (tuple): (height, width) of output image. Auto-detected if None.
        filter_type (str): Filter type ('ramp', 'hann' or None)
        cutoff (float): Frequency cutoff (0-1)
        
    Returns:
        np.ndarray: Reconstructed image
    """
    num_angles, num_detectors = sinogram.shape
    
    # Auto-detect image size if not provided
    if image_size is None:
        diag = int(np.ceil(num_detectors / np.sqrt(2)))
        image_size = (diag, diag)
    
    # 1. Apply frequency domain filtering
    filtered_sino = np.zeros_like(sinogram)
    for i in range(num_angles):
        filtered_sino[i, :] = apply_filter(sinogram[i, :], filter_type, cutoff)
    
    # 2. Backprojection
    recon = np.zeros(image_size)
    center = np.array(image_size) // 2
    y, x = np.indices(image_size)
    x = x - center[1]
    y = y - center[0]
    
    for angle_idx, angle in enumerate(theta):
        theta_rad = np.deg2rad(angle)
        
        # Calculate detector positions for this angle
        detector_pos = x * np.cos(theta_rad) + y * np.sin(theta_rad)
        
        # Map to detector indices
        detector_idx = (detector_pos + num_detectors/2) * (num_detectors-1)/num_detectors
        #detector_idx = detector_pos + (num_detectors - 1) / 2
        
        # Create interpolation function for this projection
        interp_fn = interp1d(np.arange(num_detectors), 
                            filtered_sino[angle_idx, :],
                            kind='linear',
                            bounds_error=False,
                            fill_value=0)
        
        # Interpolate and accumulate
        recon += interp_fn(detector_idx)

    # Normalization
    return recon * np.pi / (2*num_angles) # do we need to remove factor 2?

# Helper function from previous implementation
def apply_filter(projection, filter_type='ramp', cutoff=1.0):
    n = len(projection)
    fourier = np.fft.fft(projection)
    freq = np.fft.fftfreq(n)
    
    if filter_type == 'ramp':
        filt = np.abs(freq)
    elif filter_type == 'hann':
        filt = np.abs(freq) * (0.5 + 0.5 * np.cos(np.pi * freq / cutoff))
    elif filter_type == None:
        filt = np.ones(len(freq))
    else:
        raise ValueError(f"Unknown filter: {filter_type}")
    
    filt[freq > cutoff] = 0
    filt[freq < -cutoff] = 0
    
    return np.fft.ifft(fourier * filt).real

sinogram = np.load("data/sinogram.npy")
theta = np.linspace(0., 180., sinogram.shape[0], endpoint=False)  # Angles from 0 to 180 degrees
print("Number of angles:",len(theta))

# Reconstruct the image using the inverse Radon transform
reconstructed_image = inverse_radon(sinogram, theta, filter_type='ramp')

num_detectors = sinogram.shape[1]

# Load the sinogram
plt.imshow(sinogram, cmap='gray', aspect='auto',extent=[0, num_detectors, 0, 180])
plt.title('Sinogram')
plt.xlabel(r"Detector position [pixels]")
plt.ylabel(r"Projection angle [deg °]")
plt.colorbar(label="Accumulated Density")
plt.tight_layout()
plt.show()

# Plot the reconstructed image
plt.imshow(reconstructed_image, cmap='gray')
plt.title("Reconstructed Image (Ramp Filtered)")
plt.axis('off')
plt.tight_layout()
plt.show()

# Using downsampled sinogram
sinogram_fast = sinogram[::4,:].copy()
theta_fast = np.linspace(0., 180., sinogram_fast.shape[0], endpoint=False)
print("Number of angles (sinogram fast):",len(theta_fast))

# Reconstruct the image using the inverse Radon transform
recon_image_downsample = inverse_radon(sinogram_fast, theta_fast, filter_type='ramp')

# Plot the reconstructed image
plt.imshow(recon_image_downsample, cmap='gray')
plt.title("Reconstructed Image from Downsampled Sinogram")
plt.axis('off')
plt.tight_layout()
plt.show()

#Figure 1: Ramp vs Hann (cutoff=0.15)

recon_ramp_020 = inverse_radon(sinogram_fast, theta_fast,
                               filter_type='ramp', cutoff=0.2)

recon_ramp_015 = inverse_radon(sinogram_fast, theta_fast,
                               filter_type='ramp', cutoff=0.15)

recon_hann_015 = inverse_radon(sinogram_fast, theta_fast,
                               filter_type='hann', cutoff=0.15)

images = [recon_ramp_020, recon_ramp_015, recon_hann_015]

vmin = min(img.min() for img in images)
vmax = max(img.max() for img in images)

fig, axs = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)

im0 = axs[0].imshow(recon_ramp_020, cmap='gray', vmin=vmin, vmax=vmax)
axs[0].set_title("Ramp filter\ncutoff = 0.20")
axs[0].axis("off")

im1 = axs[1].imshow(recon_ramp_015, cmap='gray', vmin=vmin, vmax=vmax)
axs[1].set_title("Ramp filter\ncutoff = 0.15")
axs[1].axis("off")

im2 = axs[2].imshow(recon_hann_015, cmap='gray', vmin=vmin, vmax=vmax)
axs[2].set_title("Hann filter\ncutoff = 0.15")
axs[2].axis("off")

plt.show()

# Figure 2: Hann filter with different cutoffs

recon_hann_1 = inverse_radon(sinogram_fast, theta_fast,
                             filter_type='hann', cutoff=1)

recon_hann_02 = inverse_radon(sinogram_fast, theta_fast,
                              filter_type='hann', cutoff=0.2)

recon_hann_03 = inverse_radon(sinogram_fast, theta_fast,
                              filter_type='hann', cutoff=0.3)

recon_hann_04 = inverse_radon(sinogram_fast, theta_fast,
                             filter_type='hann', cutoff=0.4)

images = [recon_hann_1, recon_hann_02, recon_hann_03,recon_hann_04]

vmin = min(img.min() for img in images)
vmax = max(img.max() for img in images)

fig, axs = plt.subplots(2, 2, figsize=(10,10), constrained_layout=True)

im0 = axs[0, 0].imshow(recon_hann_1, cmap='gray', vmin=vmin, vmax=vmax)
axs[0, 0].set_title("Hann filter\ncutoff = 1.0")
axs[0, 0].axis("off")

im1 = axs[0, 1].imshow(recon_hann_04, cmap='gray', vmin=vmin, vmax=vmax)
axs[0, 1].set_title("Hann filter\ncutoff = 0.4")
axs[0, 1].axis("off")

im2 = axs[1, 0].imshow(recon_hann_03, cmap='gray', vmin=vmin, vmax=vmax)
axs[1, 0].set_title("Hann filter\ncutoff = 0.3")
axs[1, 0].axis("off")

im3 = axs[1, 1].imshow(recon_hann_02, cmap='gray', vmin=vmin, vmax=vmax)
axs[1, 1].set_title("Hann filter\ncutoff = 0.2")
axs[1, 1].axis("off")

plt.show()
