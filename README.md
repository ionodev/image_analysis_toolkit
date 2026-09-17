# Image Analysis Toolkit

Image analysis scripts: Fourier-domain analysis, image registration,
multispectral remote sensing, and CT/3D reconstruction.

## Layout

| Script | Description |
|---|---|
| [`fourier_spectrum_analysis.py`](fourier_spectrum_analysis.py) | 2D FFT magnitude/phase spectra and histogram comparison across four test images. |
| [`phase_orientation_analysis.py`](phase_orientation_analysis.py) | Local phase/orientation analysis via structure tensor and Laplacian gradients on a brochure image. |
| [`landsat_band_composites.py`](landsat_band_composites.py) | Multispectral Landsat band visualization and false-color composites. |
| [`ct_sinogram_reconstruction.py`](ct_sinogram_reconstruction.py) | Inverse Radon transform CT reconstruction from a sinogram, comparing ramp and Hann filters. |
| [`hydrogen_orbital_3d_reconstruction.py`](hydrogen_orbital_3d_reconstruction.py) | 3D reconstruction of a hydrogen orbital from noisy volumetric data, evaluated with PSNR/SSIM. |
| [`phase_correlation_template_matching.py`](phase_correlation_template_matching.py) | Template localization in an image via phase correlation. |

## Data requirements

All input data (`data/`) is included directly in the repo — source images,
the Landsat `.mat` cube, the CT `sinogram.npy`, and the hydrogen orbital
`.vtk` mesh.

## Requirements

Scripts use `numpy`, `scipy`, `matplotlib`, `scikit-image`; `pyvista` and
`Pillow` (3D hydrogen reconstruction).
