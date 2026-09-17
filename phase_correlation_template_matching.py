from skimage import io, img_as_float64
from skimage.transform import rotate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def phase_correlation_template_match(search_img, template_array):
    """
    Locate a single template in a search image using phase correlation.

    Parameters:
        search_img (ndarray): Preloaded search image (float64 grayscale).
        template_array (ndarray): Template image array (float64 grayscale).

    Returns:
        match_info (tuple): (x, y, w, h) location and size of template match.
    """
    # Get template size
    h, w = template_array.shape

    # Pad template to the same size as search image
    padded_template = np.zeros_like(search_img)
    padded_template[:h, :w] = template_array

    # FFT of search image and template
    F_search = np.fft.fft2(search_img)
    F_template = np.fft.fft2(padded_template)

    # Compute cross-power spectrum
    R = F_search * np.conj(F_template)
    R /= np.abs(R) + 1e-8  # avoid division by zero

    # Inverse FFT to get correlation map
    corr = np.fft.ifft2(R)
    corr = np.abs(corr)

    # Find peak location
    y, x = np.unravel_index(np.argmax(corr), corr.shape)

    # Adjust for FFT wrap-around
    if x > search_img.shape[1] // 2:
        x -= search_img.shape[1]
    if y > search_img.shape[0] // 2:
        y -= search_img.shape[0]

    return (x, y, w, h)


def visualize_match(search_img, match_info, template_name, x_txt=0, y_txt=0):
    """
    Visualize template match with label.

    Parameters:
        search_img (ndarray): Search image.
        match_info (tuple): (x, y, w, h)
        template_name (str): Name to display on the image.
    """
    x, y, w, h = match_info

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(search_img, cmap='gray')

    rect = patches.Rectangle((x, y), w, h,
                             linewidth=2,
                             edgecolor='red',
                             facecolor='none')
    ax.add_patch(rect)

    ax.text(x+x_txt, y+y_txt,
            template_name,
            color='red',
            fontsize=12,
            weight='bold')

    ax.axis('off')
    plt.tight_layout()
    plt.show()


# Preload search image
search_image = img_as_float64(io.imread("data/search.png", as_gray=True))

# Load templates
template1 = img_as_float64(io.imread("data/template1.png", as_gray=True))
template2 = img_as_float64(io.imread("data/template2.png", as_gray=True))

# rotate template2 45 deg CCW to match search image
template2_rotated = rotate(template2, angle=45, resize=True)

# Match template1
match1 = phase_correlation_template_match(search_image, template1)
visualize_match(search_image, match1, 'template_1', y_txt=-5)

# Match rotated template2
match2 = phase_correlation_template_match(search_image, template2_rotated)
visualize_match(search_image, match2, 'template_2', x_txt=140, y_txt=450)