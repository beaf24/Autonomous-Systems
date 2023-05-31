from PIL import Image

# Open the PGM file
image = Image.open("Com_mapping.pgm")

# Get the pixel data of the image
pixel_data = image.load()

# Create a text file for writing the gray level values
output_file = open("output.txt", "w")

# Iterate over each pixel in the image
for y in range(image.height):
    for x in range(image.width):
        # Get the grayscale value of the pixel
        gray_level = pixel_data[x, y]

        # Write the grayscale value to the text file
        output_file.write(f"Pixel at ({x}, {y}): Gray level = {gray_level}\n")

# Close the output file
output_file.close()
