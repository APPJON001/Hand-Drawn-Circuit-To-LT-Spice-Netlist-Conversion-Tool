
'''Find the closest and furthest terminal to the centroid of some component'''
def find_component_orientation(original_image, xyxy, terminals):
    component_image = extract(original_image, xyxy) # Extract component from original image
    Pb = get_black_pixels(component_image) # Get a list of black pixels
    # Determine centroids
    sum_x, sum_y = None, None
    for x, y in black_pixels:
        sum_x += x; sum_y += y;
    cx = sum_x/len(Pb); cy = sum_y/len(Pb)
    # Map centroid coordinates back to the original image
    og_cx = xyxy[0] + cx; og_cy = xyxy[2] + cy
    # Determine which terminal is closer based on Euclidean distance
    terminal_1, terminal_2 = terminals
    distance_to_t1 = get_dist(og_cx, og_cy, terminal_1)
    distance_to_t2 = get_dist(og_cx, og_cy, terminal_2)
    # Return closest terminal, furthest terminal
    if terminal_1 > terminal_2: return distance_to_t1, distance_to_t2
    else: return distance_to_t2, distance_to_t1

    
    
    
    
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