'''
The pre-processing class that contains functions to pre-process the pipeline's input images.
'''

import cv2
from PIL import Image
import numpy as np

class preProcess:

    @staticmethod
    def global_binarize(input_path, output_path, threshold_value=100, kernel_size=3):
        try:
            # Read the input image using OpenCV
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply global thresholding
            _, binarized_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Define a kernel for morphology operations (adjust size as needed)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Perform noise reduction using morphological operations (erosion and dilation)
            binarized_image = cv2.erode(binarized_image, kernel, iterations=1)
            binarized_image = cv2.dilate(binarized_image, kernel, iterations=1)
            
            # Save the processed image to the output path
            cv2.imwrite(output_path, binarized_image)
            
            print(f"Binarization and noise reduction complete for {output_path}")
            
        except Exception as e:
            print(f"An error occurred for {input_path}: {e}")
        
        
    @staticmethod
    def local_binarize(input_path, output_path, block_size=21, constant=10, kernel_size=3):
        try:
            # Read the input image using OpenCV
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply adaptive thresholding
            binarized_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, block_size, constant)
            
            # Define a kernel for morphology operations (adjust size as needed)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Perform noise reduction using morphological operations (erosion and dilation)
            binarized_image = cv2.erode(binarized_image, kernel, iterations=1)
            binarized_image = cv2.dilate(binarized_image, kernel, iterations=1)
            
            # Save the binarized image
            cv2.imwrite(output_path, binarized_image)
            
            print("Binarization, erosion, and dilation complete.")
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_image_path = "./test.png"
    output_image_path_global = "test_binarized_global.png"
    output_image_path_local = "test_binarized_local.png"
    block_size_value = 11  # Adjust as needed
    constant_value = 10   # Adjust as needed
    threshold_value = 128  # Adjust as needed
    kernel_size = 3       # Adjust as needed
    
    preProcess.global_binarize(input_image_path, output_image_path_global, threshold_value, kernel_size)
    preProcess.local_binarize(input_image_path, output_image_path_local, block_size_value, constant_value, kernel_size)