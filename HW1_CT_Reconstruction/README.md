=============================
# Computed Tomography (CT) Reconstruction
=============================

This repository contains a Python implementation of CT reconstruction techniques, including filtered back projection, unfiltered back projection, and an algebraic iterative reconstruction method (SIRT). The code demonstrates how varying the number of projection angles and using different reconstruction techniques can affect the quality of the reconstructed image. The Shepp-Logan phantom, a widely-used test image for CT, is used as the input image.

## Dependencies
============
- NumPy
- Matplotlib
- scikit-image
- SciPy

## Running the Code
================

1. Ensure all dependencies are installed in your Python environment.
2. Run the Python script, which will execute the CT reconstruction techniques and display the results.

## Implementation Details
======================

1. The Shepp-Logan phantom is created using the `ascent` function from SciPy datasets.
2. A sinogram is generated using the Radon Transform from the scikit-image library.
3. Different numbers of projection angles (18, 24, and 90) are used to reconstruct the images.
4. The differences between filtered and unfiltered back projections are demonstrated by plotting the reconstructed images.
5. The SIRT (Simultaneous Iterative Reconstruction Technique) method from the scikit-image library is used as an algebraic iterative reconstruction technique. The reconstructed image is plotted at different iterations.

## Results
=======

The code will generate a series of plots, including:

1. The Shepp-Logan phantom
2. The sinogram of the phantom
3. Reconstructed images with varying projection angles (18, 24, and 90)
4. Filtered and unfiltered back projections
5. SIRT reconstructions at different iterations

These plots help demonstrate the effects of varying projection angles and reconstruction techniques on the quality of the reconstructed image in CT.
