# 1D PIC Plasma Simulation

This repository contains Python scripts for simulating plasma dynamics using the Particle-In-Cell (PIC) method in a one-dimensional electrostatic framework. The project is based on the work of Gomez, Hoyos, and Valdivia, and includes additional analyses of Langmuir waves and plasma instabilities.

## Project Overview

The Particle-In-Cell method is a widely-used numerical technique for simulating the behavior of plasmas. This repository implements the PIC method in one dimension, focusing on:

- Plasma dynamics with two particle beams, initialized from a Maxwell-Boltzmann distribution.
- Calculation of charge density and electric fields using the Nearest Grid Point (NGP) method.
- Numerical solution of Poisson’s equation for electric fields.
- Time evolution of the system using the Euler method.
- Analysis of Langmuir waves and comparison with theoretical dispersion relations.

## Features

- **1D PIC Simulation:** Simulates two-stream instability with Maxwell-Boltzmann velocity distribution.
- **Langmuir Waves Analysis:** Multiprocessing script for simulating Langmuir waves and calculating their dispersion relation.
- **Energy Conservation:** Tracks kinetic and potential energies over time to ensure simulation stability.

## Requirements

This project requires Python 3 and the following libraries:

- `numpy`
- `scipy`
- `matplotlib`

You can install the dependencies using pip:

```bash
pip install numpy scipy matplotlib
```
## Usage
1. **PIC_simulation_1D.py:** This script runs the main PIC simulation of plasma dynamics in one dimension. It initializes two particle beams, calculates charge densities and electric fields, and tracks the time evolution of the system.

Run the simulation with:

```bash
python PIC_simulation_1D.py
```

2. **Langmuir_waves_multiproc.py:** This script focuses on simulating Langmuir waves using multiprocessing for improved performance. It analyzes the dispersion relation and compares the results with theoretical predictions.

Run the simulation with:

```bash
python Langmuir_waves_multiproc.py
```
## Simulation Parameters
In both scripts, you can adjust the simulation parameters such as:

- `L` (System Length)
- `Ng` (Number of Grid Cells)
- `N1`, `N2` (Number of Particles in each beam)
- `Nt` (Number of Time Steps)
- `v0` (Initial Velocities of the beams)
- `vth` (Thermal Velocity)
These parameters are defined at the beginning of each script and can be customized to explore different plasma configurations.

## Results
<img width="1004" alt="image" src="https://github.com/user-attachments/assets/1b22b098-4ace-4f02-9cd2-8b35449b5edc">
<img width="1004" alt="image" src="https://github.com/user-attachments/assets/4de335b6-f822-4be8-b2ef-675893f507f7">
<img width="1004" alt="image" src="https://github.com/user-attachments/assets/56d4f7da-7807-41b4-b141-e176cbbb5e64">

## References
1. Gomez, S., Hoyos, J. H., & Valdivia, J. A. (2023). Particle-in-cell method for plasmas in the one-dimensional electrostatic limit. American Journal of Physics, 91(3), 225-230. doi:10.1119/5.0135515
2. Cowan, G. (1998). Statistical Data Analysis. Oxford University Press.
