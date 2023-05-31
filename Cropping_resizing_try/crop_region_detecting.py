import cv2
import numpy as np
from PIL import Image


#To crop out the similar parts from two images and return them as NumPy arrays
def crop_and_resize_images(pgm_image_path, png_image_path, target_size):
    # Load the images
    img_pgm = cv2.imread(pgm_image_path, cv2.IMREAD_GRAYSCALE)
    img_png = cv2.imread(png_image_path, cv2.IMREAD_UNCHANGED)

    # Threshold the pgm image to extract corridors
    _, thresholded = cv2.threshold(img_pgm, 210, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (which should correspond to the corridors)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    mask = np.zeros_like(img_pgm)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Resize the mask to match the dimensions of the png image
    mask_resized = cv2.resize(mask, (img_png.shape[1], img_png.shape[0]))

    # Create an alpha channel based on the mask
    alpha_channel = np.zeros_like(img_png[:, :, 3])
    alpha_channel[np.where(mask_resized == 255)] = 255

    # Apply the alpha channel to the png image
    img_png[:, :, 3] = alpha_channel

    # Resize both images to the target size
    img_pgm_resized = cv2.resize(img_pgm, target_size)
    img_png_resized = cv2.resize(img_png, target_size)

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







