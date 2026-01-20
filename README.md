NOT UPTATED...CONTACT ME FOR THE MOST RECENT VERSION

# cuULDM

GPU (CUDA) simulator for ultralight dark matter (ULDM) dynamics based on the Schrödinger–Poisson system.  
Implements a 3D pseudo-spectral solver with split-step (Strang splitting) integration and Poisson equation solved in Fourier space.

## ✨ Features

- Numerical evolution of the Schrödinger–Poisson equations:
  \[
    i \partial_t \psi = -\frac{1}{2}\nabla^2 \psi + \phi \psi, \quad \nabla^2 \phi = 4 \pi |\psi|^2
  \]
- **Pseudo-spectral method** using 3D FFTs (cuFFT).  
- **Split-step scheme** (half-step potential + full-step kinetic).  
- Poisson solver in Fourier space.  
- Initialization with **tabulated soliton profile** or optional point-mass contribution.  
- Direct visualization. 

## 🚀 Requirements

- NVIDIA GPU 
- **CUDA Toolkit** (>= 11.x recommended)  
- **cuFFT** (bundled with CUDA Toolkit)  
- `nvcc` compiler  

## ⚙️ Build

Clone the repository and compile with:

```bash
nvcc -O3 -arch=sm_75 cuULDM.cu -lcufft -o cuULDM
