import cv2 # Import cv2 for canny edge detector


def canny_edge(input_image = '../data/3b.png'):
    
    # Load your image
    image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale

    # Apply Gaussian blur to reduce noise (adjust the kernel size as needed)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred_image, threshold1=30, threshold2=100)  # Adjust thresholds as needed

    # Invert the colors (white background, black foreground)
    inverted_edges = cv2.bitwise_not(edges)

    # Save or display the resulting edges
    cv2.imwrite('../data/3edges.png', inverted_edges)  # Save the inverted edge image
    
    return inverted_edges # Return the inverted edges image


# Call canny_edge detector by default
if __name__ == "__main__":
    canny_edge()