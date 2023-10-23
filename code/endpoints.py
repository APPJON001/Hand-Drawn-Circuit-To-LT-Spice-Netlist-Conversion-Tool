import random
import time
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw



class endpoints:
    def __init__(self, image):
        self.image = image        
         
    def get_nearest_pixel(self, current_pixel, threshold):
        loaded_image = self.image.load()
        x, y = current_pixel
        width, height = self.image.size

        # Define the search box size
        box_size = 1

        while box_size <= threshold:
            print("box_size = ", box_size)
            # Iterate along the top and bottom borders
            for i in range(x - box_size, x + box_size + 1):
                for j in [y - box_size, y + box_size]:
                    if 0 <= i < width and 0 <= j < height and loaded_image[i, j] == 0:
                        print("returning pixel: ", [i, j])
                        self.image.putpixel(current_pixel, 255) # Put white pixel over current pixel
                        return [i, j]

            # Iterate along the left and right borders (excluding corners to avoid duplication)
            for i in [x - box_size, x + box_size]:
                for j in range(y - box_size + 1, y + box_size):
                    if 0 <= i < width and 0 <= j < height and loaded_image[i, j] == 0:
                        print("returning pixel: ", [i, j])
                        self.image.putpixel(current_pixel, 255) # Put white pixel over current pixel

                        return [i, j]

            # Expand the search box size for the next iteration
            box_size += 1

        # If no black pixel is found within the threshold, return None
        return None


    '''Takes in a pixels coordinates and returns the closest terminal coordinates and the distance to it'''
    @staticmethod
    def check_pixel_terminal_dist(current_pixel, terminal_list):
        closest_terminal = None
        closest_terminal_dist = float('inf')  # Initialize with a large value
        
        for terminal in terminal_list:
            terminal_x, terminal_y = terminal
            pixel_x, pixel_y = current_pixel
            
            # Calculate Euclidean distance
            distance = math.sqrt((terminal_x - pixel_x)**2 + (terminal_y - pixel_y)**2)
            
            # Update closest_terminal and closest_terminal_dist if this terminal is closer
            if distance < closest_terminal_dist:
                closest_terminal_dist = distance
                closest_terminal = terminal
        
        return closest_terminal_dist, closest_terminal
        
        
    # Runs the connection location algorithm to determine component connections.
    @staticmethod
    def run_algorithm(image, P):
        
        i = 0 # Index the start endpoint
        
        for x, y in P: # For every line terminal point.
            
            current_pixel = (x, y) # Store current pixel
            #connected_enpoint_found = False # We have not found a connected endpoint.
            print("pixel = ", current_pixel)
            while i < 6000: # Trace pixels until we've found a connected endpoint.
                
                nearest_pixel = endpoints.get_nearest_pixel(image, current_pixel, 10) # Get the nearest pixel to the current pixel.
                #print("pixel = ", nearest_pixel)
                image.putpixel(current_pixel, 255) # Put white pixel over current pixel.
                current_pixel = nearest_pixel # Update current_pixel.
                
                #time.sleep(1)  # Delay for 300ms
                if i == 5000:
                    image.show()  # Show the updated image

                current_pixel = nearest_pixel
                
                i += 1 # Move on to consider next endpoint.
        return 0

    '''Determine if a considered pixel is in a list of pixels'''
    @staticmethod
    def is_in_array(pixel, pixel_list):
        for logged_endpoint in pixel_list:
            if pixel[0] == logged_endpoint[0] and pixel[1] == logged_endpoint[1]: 
                return True
        return False

    # Put pixels at certain locations in the image - FOR TESTING PURPOSES
    @staticmethod
    def put_pixels(image, P, value):
        modified_image = image.copy() # Create a copy of the input image
        for x, y in P:
            modified_image.putpixel((x, y), value) # Set the pixel value at (x, y) to gray (128)
        return modified_image


    # Count the number of gray pixels in a grayscale image (not black, not white)
    @staticmethod
    def count_non_black_non_white_pixels(grayscale_image):
        count = 0
        width, height = grayscale_image.size
        
        for x in range(width):
            for y in range(height):
                pixel_value = grayscale_image.getpixel((x, y))
                if pixel_value > 0 and pixel_value < 255:  # Check if pixel is neither black nor white
                    count += 1
                    print("x, y = ", x, y)
        return count


    # Find the nearest black pixel to a given pixel
    @staticmethod
    def get_nearest_black_pixel(image_array, current_pixel):
        black_pixels = np.column_stack(np.where(image_array == 0))
        distances = np.sqrt((black_pixels[:, 0] - current_pixel[0]) ** 2 + (black_pixels[:, 1] - current_pixel[1]) ** 2)
        nearest_index = np.argmin(distances)
        return black_pixels[nearest_index]
    
        
if __name__ == "__main__":
    
    input_image_path = "../data/3edges.png" # Load in image
    image = Image.open(input_image_path).convert("L") # Convert to grayscale
    print("Image width, height = ", image.size)
    # Array of endpoint tuples
    #P = [(50, 300), (63, 234), (506, 240), (508, 327), (312, 255), (310, 280), (202, 715), (268, 712)]
    P = [(400, 400)]
    modified_image = endpoints.put_pixels(image, P, 128) # Define endpoints
    nearest_points = endpoints.run_algorithm(modified_image, P) # Define endpoints 
    print(nearest_points) # Returns nearest pixels to those in P. 