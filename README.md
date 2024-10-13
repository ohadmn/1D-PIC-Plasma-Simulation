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
