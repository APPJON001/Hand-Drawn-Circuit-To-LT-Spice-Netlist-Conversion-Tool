'''FIX THIS - OUT OF BOUNDS NOT CORRECT - SO IT SEEMS!!!'''
    
# Implement an outgoing square search to find nearest black pixel.
def get_nearest_pixel(self, current_pixel, threshold): 
    
    width, height = self.image.size # Get the width and height of the image in pixels
    
    padding_dist = 1 # Distance around the pixel that we are searching at
    
    x, y = current_pixel # Define current pixel location
    try:
        while True:
            
            square_side = 2*padding_dist + 1 # Side length of considered square
            
            # If box is totally outside image range, all pixels in image have been searched, there is nothing more to do
            if (x - padding_dist) < 0 and (y - padding_dist) < 0 and (x + padding_dist) > width and (y + padding_dist) > height:
                print("Pixel search went out of bounds!")
                return None
            
            else:            
                # Examine box around current pixel
                for dx in range(square_side - 1):
                    if ((x - padding_dist + dx >= 0) and (x - padding_dist + dx <= width) and (y - padding_dist >= 0) and (y - padding_dist <= height)):
                        val = self.image.getpixel((x - padding_dist + dx, y - padding_dist)) # Top left to 1 before top right
                        if val < threshold: # Part of a connection line
                            return (x - padding_dist + dx, y - padding_dist)
                        
                for dy in range(square_side - 1):
                    if ((x + padding_dist >= 0) and (x + padding_dist <= width) and (y - padding_dist + dy >= 0) and (y - padding_dist + dy <= height)):
                        val = self.image.getpixel((x + padding_dist, y - padding_dist + dy)) # Top right to 1 before bot right
                        if val < threshold: # Part of a connection line
                            return (x + padding_dist, y - padding_dist + dy)
                    
                for dx in range(square_side - 1):
                    if ((x - padding_dist + dx + 1 >= 0) and (x - padding_dist + dx + 1 <= width) and (y + padding_dist >= 0) and (y + padding_dist <= height)):
                        val = self.image.getpixel((x - padding_dist + dx + 1, y + padding_dist)) # Top right to 1 before bot left
                        if val < threshold: # Part of a connection line
                            return (x - padding_dist + dx + 1, y + padding_dist)
                    
                for dy in range(square_side - 1):                    
                    if ((x - padding_dist >= 0) and (x - padding_dist <= width) and (y - padding_dist + dy + 1 >= 0) and (y - padding_dist + dy + 1 <= height)):
                        val = self.image.getpixel((x - padding_dist, y - padding_dist + dy + 1)) # Top left to 1 before top right
                        if val < threshold: # Part of a connection line
                            return (x - padding_dist, y - padding_dist + dy + 1)
                    
                padding_dist += 1 # If no connection pixels are in box, consider next box out
    except: 
        print("Pixel search went out of bounds!")
        exit()