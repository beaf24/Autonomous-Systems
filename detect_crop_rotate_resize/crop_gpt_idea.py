from PIL import Image

def extract_corridor(image_path, corridor_color=(250, 250, 250), threshold=210):
    # Open the image
    image = Image.open(image_path)
    
    # Convert the image to RGB format if it's in grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Find the bounding box of the corridors
    width, height = image.size
    left = width
    right = 0
    top = height
    bottom = 0
    
    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))
            if pixel == corridor_color or sum(pixel) > threshold:
                # Update the bounding box coordinates
                left = min(left, x)
                right = max(right, x)
                top = min(top, y)
                bottom = max(bottom, y)
    
    # Crop the image using the calculated bounding box
    cropped_image = image.crop((left, top, right, bottom))
    
    return cropped_image

# Example usage
pgm_image_path = 'Com_gmapping.pgm'
cropped_image = extract_corridor(pgm_image_path)

# Save the extracted corridor image
cropped_image.save('cropped_image.png')
