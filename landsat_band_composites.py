import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float64
from scipy.io import loadmat

landsat_data = loadmat('data/landsat_data.mat')
data = landsat_data['landsat_data']

def plot_data(invert=False):
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))

    for i in range(7):
        row = i // 4
        col = i % 4

        img = img_as_float64(data[:, :, i])

        if invert:
            ax[row, col].imshow(img, cmap='gray_r')
        else:
            ax[row, col].imshow(img, cmap='gray')
        ax[row, col].set_title(f'Band {i+1}')
        ax[row, col].axis('off')

    ax[1, 3].axis('off')

    plt.tight_layout()
    plt.show()

plot_data()
plot_data(invert=True)

rows, cols, _ = data.shape

def invert_band(band):
    # Helper function to invert band
    return 1 - img_as_float64(data[:, :, band])


# Define alternative bands for the R (red) and G (green) channel
R_bands = [2, 6]  # Band 3, Band 7
G_options = [3, ('invert', 1), ('invert', 2), ('invert', 6)]  # Band 4, inv 2, inv 3, inv 7
B_band = np.zeros((rows, cols))

# Generate composites and titles
composites = []
titles = []

for R in R_bands:
    for G in G_options:
        # Get R channel
        R_data = img_as_float64(data[:, :, R])

        # Get G channel
        if isinstance(G, tuple) and G[0] == 'invert':
            G_data = invert_band(G[1])
            G_title = f"{G[1] + 1}^{{-1}}"
        else:
            G_data = img_as_float64(data[:, :, G])
            G_title = f"{G + 1}"

        # Compose RGB
        rgb = np.dstack((R_data, G_data, B_band))
        composites.append(rgb)

        # Title as (R,G,B)
        titles.append(rf'$({R + 1}, {G_title}, 0)$')


# This is the normal RGB composite using the Red, Green and Blue bands
rgb_urban_visible = np.dstack(
    (img_as_float64(data[:, :, 2]), img_as_float64(data[:, :, 1]), img_as_float64(data[:, :, 0])))

composites.extend([rgb_urban_visible])
titles.extend([r'$(3,2,1)$'])

# Plot all composites in a grid
n = len(composites)
rows_plot = 3
cols_plot = int(np.ceil(n / rows_plot))

fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(11, 11))
axes = axes.ravel()

for i, ax in enumerate(axes):
    if i < n:
        ax.imshow(composites[i])
        ax.set_title(titles[i], fontsize=10)
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()

# Best candidates
rgb_best_1 = np.dstack((img_as_float64(data[:,:,2]), img_as_float64(data[:,:,3]), np.zeros(np.shape(data[:,:,0]))))
rgb_best_2 = np.dstack((img_as_float64(data[:,:,6]), img_as_float64(data[:,:,3]), np.zeros(np.shape(data[:,:,0]))))

fig, ax = plt.subplots(1, 2,figsize=(12, 7))

# Plot the RGB images
ax[0].imshow(rgb_best_1)
ax[0].set_title('Bands: (3, 4, 0)')
ax[0].axis('off')
ax[1].imshow(rgb_best_2)
ax[1].set_title('Bands: (7, 4, 0)')
ax[1].axis('off')

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(2, 2, figsize=(11, 11))

# Band indices for the blue channel
indices = [3, 4, 5, 6]

for idx, i in enumerate(indices):
    rgb_best_overall = np.dstack((
        img_as_float64(data[:, :, 2]),  # Red channel
        img_as_float64(data[:, :, 3]),  # Green channel
        1 - img_as_float64(data[:, :, i])  # Blue channel (inverted)
    ))

    # Determine subplot position
    row = idx // 2
    col = idx % 2

    ax[row, col].imshow(rgb_best_overall)
    ax[row, col].set_title(f'(3, 4, ${i+1}^{{-1}}$)')
    ax[row, col].axis('off')

plt.tight_layout()
plt.show()