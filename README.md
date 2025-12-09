##Quantum Window Simulation

### About

The "Quantum Window" device provides visualization of the probable future state of the observed scene.
The device contains a matrix of homodyne detectors that record the quantum statistics of light without completely destroying phase information during data acquisition. 
Based on a series of measurements, the processor reconstructs the density matrices for each pixel. A neural network, trained on the history of quantum dynamics,
extrapolates the state of the density matrices to a future time interval specified by the user.
The image is formed by reading the diagonal elements (photon number distribution) of the predicted matrices.
The technical result is the ability to observe the dynamics of processes with a lead in real time by analyzing the hidden parameters of quantum evolution.

### Installation

```sh
pip install torch numpy matplotlib scipy
```

### Output

![Quantum Window Simulation](https://github.com/stowage/quantum_window/blob/main/qw_photon.png?raw=true)



