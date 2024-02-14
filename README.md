# WF-CNN
Wiener Filter trained models for CMB maps with inhomogeneous noise applied. 

Simplified demonstrantion of the Wiener Filter (WF) reconstruction for Cosmic Microwave Background (CMB) maps with inhomogeneous noise applied, using Convolutional Neural Networks (CNN). The CNN models present here are the ones trained for the paper (https://arxiv.org/abs/2312.09943). 

# Requirements: 

- Tensorflow 2.X
- CAMB (https://camb.readthedocs.io/en/latest/)

# Usage: 

The implemented codes here perform the WF reconstruction of CMB maps with inhomogeneous noise applied. The inhomogeneous noise is represented by four different variance maps, two created from linear and quadratic functions, and two examples extracted from Planck. The reconstruction utilizes the trained weights for each specific problem.

To check the results presented: 

1. Edit the input dictionary ``input_demo.dict`` with the desired example.
2. Run ``python eval_demo.py input_demo.dict`` to obtain the WF prediction with the CNN model.
3. Run ``python spectrum_script.py input_demo.dict`` to estimate the power spectrum given the filtered map.

# Contact 

Feel free to contact me at belen@fcaglp.unlp.edu.ar for comments, questions and suggestions.
