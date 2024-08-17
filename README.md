# Introduction

This repository holds the code for my honours thesis project. Please star this repository if you find the included research helpful.

Object detection with a state-of-the-art YOLOv8 model was used to detect components in a given hand drawn circuit schematic. Advanced image processing techniques (The Hough Transform, Pixel Density Analysis, Pixel Tracing, etc.) were then applied to determine how components were connected together. Information gathered from the previous stages was then used to generate a simulatable netlist in LT-Spice format. Overall, high conversion accuracies were achieved for schematics with non-terminalled components. Components that were terminaled (diodes and DC sources) were not determined accurately, since minimal focus was given to this subset. 

The full system pipeline is depicted graphically below:

![image](https://github.com/user-attachments/assets/51fd0fbc-6380-4613-bbac-d14bb21c4d1b)

![image](https://github.com/user-attachments/assets/bd4bb833-22aa-4b13-b064-0a5944d7ecbb)

Python, Flask, and SQLAlchemy made up the foundation of the web-application, which facilitated the entirety of the pipeline process.

# More

Please note that this repository is based on my research conducted at the University of Cape Town. I have included my thesis below for additional detail. 

[JAPPSTHESIS.pdf](https://github.com/user-attachments/files/16644417/JAPPSTHESIS.pdf)

# Installation

This repository is still under construction - not all the training data/labels are present. 
However, the web application should run once requirements are installed. 

Once the requirements are installed, the web application can be run by navigating into the web_app directory and running:

> python3 main.py

Ensure that the script listens on an unused port. Navigate to the terminal-specified page in your favourite browser to use.

Note: This code currently only works on MacOS. A Windows-compatible version should be released soon.
