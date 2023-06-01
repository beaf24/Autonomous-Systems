import cv2
import numpy as np
from PIL import Image


#To crop out the similar parts from two images and return them as NumPy arrays
import cv2

def crop_and_resize_images(pgm_image_path, png_image_path, target_size):
    # Load the images
    img_pgm = cv2.imread(pgm_image_path, cv2.IMREAD_GRAYSCALE)
    img_png = cv2.imread(png_image_path)

    # Threshold the pgm image to extract corridors
    _, thresholded = cv2.threshold(img_pgm, 205, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (which should correspond to the corridors)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    mask = cv2.drawContours(thresholded.copy(), [largest_contour], 0, (255, 255, 255), thickness=cv2.FILLED)

    # Apply the mask to the pgm image
    img_pgm_cropped = cv2.bitwise_and(img_pgm, mask)

    # Resize both images to the target size
    img_pgm_resized = cv2.resize(img_pgm_cropped, target_size)
    img_png_resized = cv2.resize(img_png, target_size)
    
    print(largest_contour)

    return img_pgm_resized, img_png_resized


#save NumPy arrays representing images as PNG files
def save_images_as_png(images, filenames):
    for image, filename in zip(images, filenames):
        # Convert the NumPy array to PIL Image
        pil_image = Image.fromarray(image)

        # Save the PIL Image as PNG
        pil_image.save(filename, "PNG")
        print(f"Saved {filename}")


pgm_image_path = "Com_gmapping.pgm"
png_image_path = "Com_algoritmo.png"

target_size = (600, 600)  # Adjust the target size as needed


# call "crop" function
pgm_cropped, png_resized = crop_and_resize_images(pgm_image_path, png_image_path, target_size)

# List of NumPy arrays representing images and corresponding filenames (to use save_images_as_png())
images = [pgm_cropped, png_resized]  # List of NumPy arrays representing images
filenames = ["pgm_cropped.png","png_resized.png"]  # List of corresponding filenames

# call "save images as PNG files" function
save_images_as_png(images, filenames)







