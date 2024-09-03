# Honours Thesis Project: Circuit Schematic Detection and Conversion

Welcome to the repository for my honours thesis project! If you find this research helpful, please consider starring this repository.

# Project Overview

This project involves detecting components in hand-drawn circuit schematics using the state-of-the-art YOLOv8 object detection model. Here’s a brief overview of the process:

* Component Detection: YOLOv8 detects various components within the circuit schematic.
* Image Processing: Advanced techniques such as Hough Transform, Pixel Density Analysis, and Pixel Tracing are used to determine how the components are connected.
* Netlist Generation: The processed information is used to create a netlist in LT-Spice format, making the schematic ready for simulation.

The system achieves high accuracy in converting schematics, especially for non-terminal components. Graphically terminalled components (like diodes and DC sources) were less accurately identified due to limited training on this subset.

The full system pipeline is depicted graphically below:

![image](https://github.com/user-attachments/assets/51fd0fbc-6380-4613-bbac-d14bb21c4d1b)

![image](https://github.com/user-attachments/assets/bd4bb833-22aa-4b13-b064-0a5944d7ecbb)

The web application is built using Python, Flask, and SQLAlchemy. These technologies support the entire pipeline process, from detection to netlist generation.

# More

Please note that this repository is based on my research conducted at the University of Cape Town. I have included my thesis below for additional detail. 

[Thesis.pdf](https://github.com/user-attachments/files/16644417/JAPPSTHESIS.pdf)

# Installation

This repository is still under construction - not all the training data/labels are present. 
However, the web application should run once requirements are installed. 

Once the requirements are installed, the web application can be run by navigating into the web_app directory and running:

> python3 main.py

Ensure that the process runs on an unused port. Navigate to the terminal-specified page in your favourite browser to access the web application.

Note: This code currently only works on MacOS. A Windows-compatible version may become available in the future.

# License

Copyright (c) 2023-2024 Jonathan Apps

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software **WITH CERTAIN RESTRICTIONS**. The Software 
may be modified for personal or corporate use, but shall not be used for
distribution, sublicensing, and/or selling. Please ensure that you make 
proper reference to this repository and its author (Jonathan Apps) whenever/wherever it is used.
Please feel free to contact me if further information is required.
