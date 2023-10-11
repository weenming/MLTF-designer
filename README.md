# Thin Film Designer

## Overview
This project is a code framework for multi-layer thin film (MLTF) design introduced in our review paper on MLTF Design.
It simulates the optical response of thin films based on the transfer matrix method and our new algorithm to both temporally and spatially efficiently evaluate the gradient. 

In this framework, the calculation of the gradient is accelerated with CUDA. The gradient of a spectrum with $1000$ wavelength points w.r.t $100$ layers' film can be evaluated in the order of $10^{-2}$ s which enables the design of complicated MLTF. Moreover, inspired by the adjoint method we alleviated the memory barrier in constructing the computation graph when scaling up to films with thousands of layers.

<!--Based on the above fundamental algorithms we implemented the classical needle design method. A new freeform design scheme is also provided, which allows the design of inhomogeneous films. Additionally, a novel thin layer removal algorithm with a lower impact on the performance is implemented. -->

The aim of this project is to

- increase the efficiency of traditional algorithms
- Search for the underlying rules determining the design results.
- find ways to better design multi-layer films.
  - lower total optical thickness
  - lower layer numbers
  - fewer "too thin" layers which are impractical in realistic manufacture.
## Usage
To get started with the Thin-Film-Design library, follow these steps:

1. Import the required classes and functions from the library:
  ```
  from optimizer import Optimizer, AdamOptimizer
  from spectrum import BaseSpectrum
  from film import FreeFormFilm, TwoMaterialFilm
  ```
2. Define your thin film stack structure and target spectra:
  ```
  film = FreeFormFilm(...)
  target_spec_ls = [BaseSpectrum(...), ...]
  ```

3. Initialize an optimizer with the film and target spectra:
  ```
  optimizer = AdamOptimizer(film, target_spec_ls, max_steps=...)
  ```
4. Run the optimization process：
  ```
  optimizer.optimize()
  ```
## Dependencies

Run on a machine with NVIDIA GPU(s) that supports CUDA.

Use `conda env create --file=environment.yml` to install dependencies. 

Note that the version of cudatoolkit should match that of the version of the driver, which can be found by the tool `nvidia-smi`

> It works out of the box for CUDA C/C++ as far as I am aware - however, because Numba doesn't know anything about forward compatibility it always tries to generate PTX for the latest version supported by the toolkit and not the driver, so the driver refuses to accept it for linking [Thread](https://github.com/numba/numba/issues/7006)


## File structure

- `script`
  - `tmm` contains functions related to TMM
    - `get_insert_jacobi.py` (deprecated) Calculate insertion Jacobi matrix for the gradient in needle method using TFNN
    - `get_jacobi.py` Calculate the Jacobi matrix in gradient descent using TFNN. Gradient w.r.t. thicknesses.
    - `get_jacobi_adjoint.py` Calculate the Jacobi matrix in gradient descent using TFNN. Backpropagation is implemented using the adjoint method. Gradient w.r.t.thicknesses.
    - `get_n.py` Calculate and set refractive indices in Film instances
    - `get_spectrum.py` Calculate spectrum from a film instance
    - `tmm_cpu`
      - archived tmm functions using cpu
  - `optimizer` implements different optimization methods
    - `LM_gradient_descent` executes gradient descent by optimizing thicknesses.
    - `adam` Adam gradient descent by optimizing thicknesses. Implemented SGD by randomly selecting both spectrum and wavelength points.
    - `needle_insert` executes the insertion process given the insertion gradient
  - `utils` contains general functions, tools for analysis, etc.
    - `get_n` Gets refractive indices of a material at specified wavelengths.
    - `loss` Implements loss functions. 
    - `substitute` Remove layers that are too thin to be practical. Adjust the thickness of adjacent layers s.t. $l_1$ deviation in $\vec{E}$ is minimized in the first-order approximation of the replaced layers being thin. 
    - `structure` function to plot the structure of a `Film` instance
  - `design.py` Implements Design objects.
  - `film.py` Implements Film objects.
  - `spectrum` Implements Spectrum objects
  
`main` files implement

- LM descent
- needle insertion iterations
- multi-thread acceleration.

`gets` module contains functions returning

- reflectance/transmittance spectrums
- gradient (Jacobi matrices) for optimizing layer thickness
- gradient for insertions in needle method.

## To-do
- parallelize inc ang
- refactor design helper in archive/LM_gradient_descent.py
- test SGD
