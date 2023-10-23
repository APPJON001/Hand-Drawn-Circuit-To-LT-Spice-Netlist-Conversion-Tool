from PIL import Image

def binarize_image(input_path, output_path, threshold=128):
    try:
        # Open the input image
        image = Image.open(input_path)
        
        # Convert the image to grayscale
        gray_image = image.convert('L')
        
        # Binarize the image using the given threshold
        binarized_image = gray_image.point(lambda p: 0 if p < threshold else 255, '1')
        
        # Save the binarized image
        binarized_image.save(output_path)
        
        print("Binarization and saving complete.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_image_path = "../data/2.png"
    output_image_path = "2global.png"
    threshold_value = 110
    binarize_image(input_image_path, output_image_path, threshold_value)