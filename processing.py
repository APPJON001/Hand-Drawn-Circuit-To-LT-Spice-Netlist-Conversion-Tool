'''
Script to remove the components from the image based on bounding box locations and the input image
'''
import cv2
import yolov8 as yl
import preprocessing as pp
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from endpoints import *

'''This class includes the image processing functions required for the system at large'''
class processing:
    
    '''Add meaningful unique identifiers to every component in a given image'''
    @staticmethod
    def add_custom_labels_to_image(image, names_dict, class_ids, xyxys):
        # Create a new dictionary with modified values
        shorthands = {'ac': 'AC', 'capacitor': 'C', 'dc': 'DC', 'diode': 'D', 'inductor': 'L', 'resistor': 'R'}
        listed_ids = []
        component_nums = []

        for id in class_ids:
            num_occurances = listed_ids.count(id) # Count the number of already-existing components for each component in class_id
            listed_ids.append(id)
            component_nums.append(num_occurances) # This specifies which number goes on which component

        # Create new confidences based on shorthands and component_nums
        new_labels = []
        for id, num in zip(class_ids, component_nums):
            shorthand = shorthands.get(names_dict.get(id, ''))
            if shorthand:
                new_label = f'{shorthand}{num}'
                new_labels.append(new_label)
            else:
                new_labels.append('')
        
        # Draw labels on the image
        draw = ImageDraw.Draw(image)  # Create a drawing context
        label_color = (30,129,176)  # Light Blue (you can adjust the color as needed)

        # Custom font style and font size
        font_path = 'Optima.ttc'  # Update with the correct font path
        font_size = 15
        myFont = ImageFont.truetype(font_path, font_size, encoding="unic")

        for (x, y, w, h), label in zip(xyxys, new_labels):
            # Draw the label on the image with bold and light blue color
            draw.text((x + 2, y), label, fill=label_color, font=myFont)

        # Save or return the modified image and labels
        return image, new_labels
    
    
    ''' Get intersection points between a given image and corresponding bounding boxes'''
    @staticmethod
    def get_imbb_intersections(image_path, xyxys):
        # Loop through the bounding boxes
        img = Image.open(image_path) # Load the preprocessed image
        loaded_img = img.load()
        intersecting_pixels = []
        for xyxy in xyxys:
            tl_x = int(xyxy[0]) # Top left x coordinate
            tl_y = int(xyxy[1]) # Top left y coordinate
            br_x = int(xyxy[2]) # Bottom right x coordinate
            br_y = int(xyxy[3]) # Bottom right y coordinate
            
            # Ensure coordinates are within bounds
            tl_x = max(tl_x, 0)
            tl_y = max(tl_y, 0)
            br_x = min(br_x, img.width - 1)
            br_y = min(br_y, img.height - 1)

            # Search in the x direction
            for i in range(br_x - tl_x + 1): # Loop through all the x pixels in the image
                
                if (loaded_img[tl_x + i, tl_y] == 0): # If the pixel in the image at this location is black
                    intersecting_pixels.append([tl_x + i, tl_y])
                if (loaded_img[tl_x + i, br_y] == 0): # If the pixel in the image at this location is black
                    intersecting_pixels.append([tl_x + i, br_y])
                        
            # Search in the y direction
            for j in range(br_y - tl_y + 1): # Loop through all the y pixels in the image
                if (loaded_img[tl_x, tl_y + j] == 0): # If the pixel in the image at this location is black
                    intersecting_pixels.append([tl_x, tl_y + j])
                if (loaded_img[br_x, tl_y + j] == 0): # If the pixel in the image at this location is black
                    intersecting_pixels.append([br_x, tl_y + j])
                    
        # Indicate and plot results
        print("intersecting_pixels = ", intersecting_pixels) # Show intersecting pixels
        white_image = Image.new('L', img.size, 255)  # Create a white image
        draw = ImageDraw.Draw(white_image) # Create a drawing context to draw on the white image
        for x, y in intersecting_pixels: draw.point((x, y), fill=0) # Set intersecting pixels to black (0)
        white_image.save('intersecting_image.png') # Save the resulting image
        # Return results
        return intersecting_pixels
    
    
    '''Method to perform K-Means searching on the grid of output pixels'''
    @staticmethod
    def k_means(intersecting_pixels, num_components):  
        print("kmeans initialized")
        k = num_components*2 # This is the number of terminals in the image!
        data = np.array(intersecting_pixels) # This is the data K-Means is performed on
        kmeans = KMeans(n_clusters=k) # Initialize K-means model
        kmeans.fit(data) # Fit the model
        terminal_locations = kmeans.cluster_centers_ # Get terminal locations
        kmeans_image = Image.open('intersecting_image.png') # Load the preprocessed image
        
        #kmeans_rgb_image = kmeans_image.convert('RGB')
        white_image = Image.new('L', kmeans_image.size, 255)  # Create a white image
        draw = ImageDraw.Draw(white_image) # Create a drawing context to draw on the white image
        for x, y in terminal_locations: draw.point((x, y), fill = 0) # Set intersecting pixels to black (0)
        white_image.save('kmeans_image.png') # Save the resulting image
        print("kmeans image saved")
        return terminal_locations
    
    
    '''Whiten out the objects'''
    @staticmethod
    def rem_components(image_path, xyxys):
        print("removing components!")
        img = Image.open(image_path)
        component_removed_img = img.copy()
        draw = ImageDraw.Draw(component_removed_img)

        for xyxy in xyxys:
            
            tl_x = int(xyxy[0]) # Top left x coordinate
            tl_y = int(xyxy[1]) # Top left y coordinate
            br_x = int(xyxy[2]) # Bottom right x coordinate
            br_y = int(xyxy[3]) # Bottom right y coordinate
            
            # Ensure coordinates are within bounds
            tl_x = max(tl_x, 0)
            tl_y = max(tl_y, 0)
            br_x = min(br_x, img.width - 1)
            br_y = min(br_y, img.height - 1)
            
            for x in range(br_x - tl_x): # Number of pixels in the x
                for y in range(br_y - tl_y): # Number of pixels in the y
                    draw.point((tl_x + x, tl_y + y), fill = 255) # Whiten out the objects
                    
        component_removed_img_path = 'component_rem_img.png' # Path to save connections image to
        component_removed_img.save(component_removed_img_path) # Save the comopnent removed image
        return component_removed_img_path # Return path to connections image
    
    
    '''Use the Canny Edge Detection Algorithm to process a given image and detect its edges'''
    @staticmethod
    def detect_edges(image_path):
         # Load your image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale

        # Apply Gaussian blur to reduce noise (adjust the kernel size as needed)
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        # Apply Canny Edge Detection
        edges = cv2.Canny(blurred_image, threshold1=30, threshold2=100)  # Adjust thresholds as needed

        # Invert the colors (white background, black foreground)
        inverted_edges = cv2.bitwise_not(edges)

        image_path = 'detected_edges.png'
        
        # Save or display the resulting edges
        cv2.imwrite(image_path, inverted_edges)  # Save the inverted edge image
        
        return image_path # Return the inverted edges image


    '''Determine how the terminals are connected, given the edge detected image and terminal locations'''
    @staticmethod
    def get_terminal_connections(edge_image_path, terminal_locations):
        terminal_locations = np.array(terminal_locations, dtype=int)

        edge_image = cv2.imread(edge_image_path, cv2.IMREAD_GRAYSCALE)
        # Given the edge image and terminal locations, determine how the terminals are connected.
        edge_detected_image = Image.fromarray(np.uint8(edge_image))

        pixelTracer = endpoints(edge_detected_image) # Object of class endpoint initialized with an edge detected image to be traced
        
        terminal_connections = {} # Stores connections to be returned  
    
        for x, y in terminal_locations: # For every line terminal point.
            print("Considering terminal [", x, "][", y, "]")

            current_terminal_connections = [] # Stores terminals connected to the current terminal
            current_pixel = [x, y] # Store current pixel
            print("pixel = ", current_pixel)
            ns_terminal_found = False # Marks if the non-starting terminal has been found or not
            i = 0 # Indexer

            while True: # Trace pixels until connections are determined
                
                distance, closest_terminal = endpoints.check_pixel_terminal_dist(current_pixel, terminal_locations)
                #print("distance, closest terminal = ", distance, ", ", closest_terminal)
                if ns_terminal_found: # If we've found a terminal we did not start from
                    if distance < 10 and (closest_terminal[0] == x and closest_terminal[1] == y): # If we've found all the connected terminals
                        edge_detected_image.show()  # Show the updated image    
                        terminal_connections[(x, y)] = current_terminal_connections # Add terminal connections to list of terminal connections
                        break
                    
                    elif distance < 10 and not endpoints.is_in_array(closest_terminal, current_terminal_connections): # Else if found a new terminal we haven't already found
                        current_terminal_connections.append(closest_terminal) # Append the closest terminal
                        
                else: # If we're on a starting terminal
                    
                    if distance < 10 and (closest_terminal[0] != x or closest_terminal[1] != y):
                        edge_detected_image.show()  # Show the updated image
                        current_terminal_connections.append(closest_terminal)                 
                        ns_terminal_found = True 
                        
                nearest_pixel = pixelTracer.get_nearest_pixel(current_pixel, 10) # Get the nearest pixel to the current pixel
                
                if nearest_pixel == None:
                    break
                edge_detected_image.putpixel(current_pixel, 255) # Put white pixel over current pixel
                current_pixel = nearest_pixel # Update current_pixel
                i += 1 # Move on to consider next endpoint    
        return terminal_connections
    
    
    '''Get the components and their associated terminals associated with each component'''
    @staticmethod
    def get_component_terminals(xyxys, component_names, terminal_locations):
        component_terminals_mapping = {}
        i = 0
        for xyxy in xyxys: # For every component outline
            # Find two closest terminals
            component_centre = [(xyxy[2] + xyxy[0])/2, (xyxy[3] + xyxy[1])/2] # Get center of component
            closest_terminals = [[], []]
            closest_distances = [float('inf'), float('inf')]
            for terminal in terminal_locations:
                distance = (((component_centre[0] - terminal[0])**2 + (component_centre[1] - terminal[1])**2)**0.5)
                # Check if the current terminal is closer than the existing ones
                if distance < closest_distances[0]:
                    closest_distances[1] = closest_distances[0]
                    closest_terminals[1] = closest_terminals[0]
                    closest_distances[0] = distance
                    closest_terminals[0] = terminal
                elif distance < closest_distances[1]:
                    closest_distances[1] = distance
                    closest_terminals[1] = terminal
            component_terminals_mapping[component_names[i]] = closest_terminals
            i += 1
        return component_terminals_mapping
    
    
    '''Plot the terminals and how they are connected to visualize terminal detection correctness'''
    @staticmethod
    def get_terminal_graph(terminal_connections, eroded_connections_path='connected_terminals.png'):
        # Open the image with PIL
        img = Image.open(eroded_connections_path)
        
        # Convert the image to RGB mode if it's not already
        img = img.convert('RGB')
        
        draw = ImageDraw.Draw(img)  # Create a drawing context

        # Define green line color
        line_color = (0, 255, 0)  # Green color

        # Draw green lines on the image
        line_thickness = 2  # Adjust line thickness as needed

        for key, values in terminal_connections.items():
            for value in values:
                # Draw a green line from key to value
                draw.line([key, value], fill=line_color, width=line_thickness)

        # Save the image with the drawn lines
        img.save('detected_connections_image.png')
    

    '''Erode the connections image for enhanced graph/pixel tracing tracking'''
    @staticmethod
    def erode(connections_img_path = 'component_rem_img.png'):
        # Load your binary image (where black lines are foreground objects)
        binary_image = cv2.imread(connections_img_path, cv2.IMREAD_GRAYSCALE)

        # Define a smaller kernel for erosion
        kernel_size = 15  # Adjust the kernel size as needed
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        eroded_image = cv2.erode(binary_image, kernel, iterations=1)
        eroded_path = 'eroded_image.png'
        # Save the eroded image to a file
        cv2.imwrite(eroded_path, eroded_image)
        return eroded_path
    
    
    ''' Determine which terminals are connected to which node'''
    @staticmethod
    def get_nodes(terminal_connections):
        print("getting nodes")
        
        terminal_connections_arr = [[list(key)] + [list(value) for value in values] for key, values in terminal_connections.items()]
        print("terminal connections array = ", terminal_connections_arr)
        
        # Create a mapping of pixels to their corresponding terminal connections
        pixel_to_terminals = {} # Stores the indicies at which pixel values are stored
        for i, terminal in enumerate(terminal_connections_arr):
            for pixel in terminal:
                if tuple(pixel) in pixel_to_terminals:
                    pixel_to_terminals[tuple(pixel)].append(i)
                else:
                    pixel_to_terminals[tuple(pixel)] = [i]

        print("pixel to terminal mapping = ", pixel_to_terminals)
        # Create a dictionary to store nodes
        nodes = {}
        visited = set() # Store the pixels that have been visited

        # Traverse the pixel_to_terminals mapping to build nodes
        for pixel, terminals in pixel_to_terminals.items(): # Pixel is the key, terminals make up the value
            if len(terminals) > 1 and tuple(pixel) not in visited:
                node_key = 'n' + str(len(nodes) + 1)
                node_terminals = []
                for terminal_index in terminals:
                    node_terminals.extend(terminal_connections_arr[terminal_index])
                    visited.add(tuple(pixel))

                nodes[node_key] = node_terminals
                
        # Filter out duplicate nodes
        updated_nodes = {}
        i = 0
        for node_id, pixel_list in nodes.items():
            if pixel_list in updated_nodes.values():
                continue
            else: 
                updated_nodes[str(i)] = pixel_list
                i += 1
                
        # Filter out duplicate values
        for node_id, pixel_list in updated_nodes.items():
            new_pixel_list = []
            for pixel in pixel_list:
                if pixel not in new_pixel_list:
                    new_pixel_list.append(pixel)
            updated_nodes[node_id] = new_pixel_list

        return updated_nodes
    
        ''' 
        # ALTERNATIVE METHOD UNFINISHED - ALSO COULD USE A LINKED LIST:
        print("getting nodes")
        terminal_connections_copy = terminal_connections
        print("terminal connections = ", terminal_connections)
        terminal_connections_arr = [[list(key)] + [list(value) for value in values] for key, values in terminal_connections.items()]
        print("terminal_connections array = ", terminal_connections_arr)
        new_components = [[],[]]
        index = 0
        for i in range(len(terminal_connections_arr)):
            for coordinate in terminal_connections[i]:
                for copied_terminals_list in terminal_connections_copy:
                    if coordinate in copied_terminals_list
                    
            delete(copied_terminals_list[i])
            index += 0
        
        # Given the terminal connections, return the node(s) that each terminal is connected to
        terminal_list = 0
        return terminal_list # Should be of form {terminal_id : [node_id_1, node_id_2, ...], ...}
        '''
        
    
    '''Determine the centroid of a DC source or diode and return the closest and furthest terminals from that centroid'''
    @staticmethod
    def find_orientation(id, xyxy, terminals):
        # Extract the component from the original image using bounding box coordinates
        original_image = cv2.imread("pp_image.png")  # Load the original image
        int_array = xyxy.astype(int)
        x1, y1, x2, y2 = int_array  # Extract bounding box coordinates
        print("x1, y1, x2, y2 = ", x1, y1, x2, y2 )
        component = original_image[y1:y2, x1:x2]  # Crop the component
    
        # Save the component as a new image
        component_name = f"component_{id}.png"
        cv2.imwrite(component_name, component)
        
        # Convert the component to grayscale
        gray_component = cv2.cvtColor(component, cv2.COLOR_BGR2GRAY)

        # Threshold the image to get black pixels
        _, thresholded = cv2.threshold(gray_component, 1, 255, cv2.THRESH_BINARY)

        # Find the centroid of black pixels
        M = cv2.moments(thresholded)
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])

        # Map centroid coordinates back to the original image
        original_centroid_x = x1 + centroid_x
        original_centroid_y = y1 + centroid_y

        # Determine which terminal is closer based on distance
        terminal_1, terminal_2 = terminals
        distance_to_1 = np.sqrt((original_centroid_x - terminal_1[0]) ** 2 + (original_centroid_y - terminal_1[1]) ** 2)
        distance_to_2 = np.sqrt((original_centroid_x - terminal_2[0]) ** 2 + (original_centroid_y - terminal_2[1]) ** 2)

        # Return the closer terminal first (logged as the positive) and the 
        if distance_to_1 < distance_to_2:
            return terminal_1, terminal_2
        else:
            return terminal_2, terminal_1
    
    
    '''Take in component terminals and return the same but with positive terminals listed first'''
    @staticmethod
    def find_pn_component_terminals(xyxys, component_terminals):
        # If component is a DC or diode, extract bounding box, get orientation, and assign terminals
        print("finding pn component terminals")
        pn_component_terminals = {}
        i = 0
        for id, terminals in component_terminals.items():
            if id[0] == 'D': # For DC sources and diodes
                closest_to_centroid_terminal, further_from_centroid_terminal = processing.find_orientation(id, xyxys[i], terminals) # Return the positive terminal
                if id[1] == 'C': # If it's a DC source
                    positive_terminal, negative_terminal = closest_to_centroid_terminal, further_from_centroid_terminal
                    pn_component_terminals[id] = [positive_terminal, negative_terminal] # Insert in positive, negative order
                else: # It's a diode
                    positive_terminal, negative_terminal = further_from_centroid_terminal, closest_to_centroid_terminal
                    pn_component_terminals[id] = [positive_terminal, negative_terminal] # Insert in positive, negative order
            else: 
                pn_component_terminals[id] = terminals # Random order is fine if not in [DC, Diode]
            i += 1
        return pn_component_terminals
        
    
    '''Generate a list of positive and negative nodes, associated with each component name'''
    @staticmethod
    def generate_component_node_mapping(pn_component_terminals, node_terminal_mapping):
        # Indicate all the values before generating netlist
        print("\nGenerating component node mapping.\n")
        
        # Function to calculate the Euclidean distance between two pixels
        def euclidean_distance(p1, p2):
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        
        pn_component_nodes = {} # This stores the positive and negative nodes of the terminals
        
        # Get the positive and negative nodes for each component
        for label, terminals in pn_component_terminals.items():
            pos_terminal, neg_terminal = terminals
            pos_node, neg_node = None, None
            min_pos_distance, min_neg_distance = float('inf'), float('inf')
            for node, node_terminals in node_terminal_mapping.items():
                #print("\npos_terminal, neg_terminal = ", pos_terminal, ", ", neg_terminal, "\n")
                #print("\nnode, node_terminals = ", node, ", ", node_terminals, "\n")
                if tuple(pos_terminal) in map(tuple, node_terminals):
                    pos_node = node
                elif tuple(neg_terminal) in map(tuple, node_terminals):
                    neg_node = node
            
            # For None resolution - Select the node that contains the closest terminal to the positive or negative terminals
            if pos_node == None:
               for node, node_terminals in node_terminal_mapping.items():
                   for node_terminal in node_terminals:
                        distance = euclidean_distance(pos_terminal, node_terminal)
                        if distance < min_pos_distance:
                            min_pos_distance = distance
                            pos_node = node
            if neg_node == None:
                for node, node_terminals in node_terminal_mapping.items():
                    for node_terminal in node_terminals:
                        distance = euclidean_distance(neg_terminal, node_terminal)
                        if distance < min_neg_distance:
                            min_neg_distance = distance
                            neg_node = node
            pn_component_nodes[label] = [pos_node, neg_node]
        
        print("\npn_component_nodes = ", pn_component_nodes)
        return pn_component_nodes
    
    
    '''This method takes in comopnent names, values, terminals, and node mappings and returns a netlist'''
    @staticmethod
    def generate_netlist(component_values, pn_component_node_mapping):
        # Indicate all the values before generating netlist
        print("\nGenerating nelist with the following data:\n")
         
        # Create the LTspice file with the .cir extension
        file_name = "netlist.cir"
    
        with open(file_name, "w") as file:
            # Write the circuit name
            file.write(f"Your Hand Drawn Circuit's Netlist\n\n")
            x = 0
            for i in range(len(component_values)):
                ac_id = ''
                name = list(pn_component_node_mapping.keys())[i]
                if name[0] == 'A': 
                    name = "V" + str(x)
                    ac_id = 'AC '
                    x += 1
                elif name[0] + name[1] == 'DC': 
                    name = "V" + str(x)
                    x += 1
                pos_node = list(pn_component_node_mapping.values())[i][0]
                neg_node = list(pn_component_node_mapping.values())[i][1]
                value = component_values[i]
                if value == 'option1': 
                    value = 'D1N4148'
                    label = ''
                elif name[0] == 'L': 
                    label = 'mH'  
                elif name[0] == 'C':
                    label = 'uF'
                else: 
                    label = ''
                
                line = f"{name} {pos_node} {neg_node} {ac_id}{value}{label}\n"
                file.write(line)
                
            file.write("\n") # Space
            file.write(".tran 0.1ms 10ms") # Transient analysis by default
            file.write("\n")

        print(f"LTspice netlist 'netlist.cir' has been generated.")
        return file_name


if __name__ == "__main__":
    print("Main")