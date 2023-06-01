import cv2
import numpy as np
from PIL import Image


#To crop out the similar parts from two images and return them as NumPy arrays
import cv2

def crop_and_resize_images(pgm_image_path, png_image_path):
    # Load the images
    img_pgm = cv2.imread(pgm_image_path, cv2.IMREAD_GRAYSCALE)
    img_png = cv2.imread(png_image_path, cv2.IMREAD_GRAYSCALE)
  

    # Threshold the pgm and png image to extract corridors
    # Find contours in the thresholded images
    _, thresholded = cv2.threshold(img_png, 205, 255, cv2.THRESH_BINARY_INV)
    contours_png, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, thresholded = cv2.threshold(img_pgm, 205, 255, cv2.THRESH_BINARY)
    contours_pgm, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Find the largest contour (which should correspond to the corridors)
    largest_contour_png = max(contours_png, key=cv2.contourArea)
    largest_contour_pgm = max(contours_pgm, key=cv2.contourArea)
 
    # Get the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour_pgm)
    a, b, c, d = cv2.boundingRect(largest_contour_png)


    print(x, y, w, h)
    print(a, b, c, d)

    # Parameter that gives "leeway" to the pgm image
    e=5
    # Parameters to compensate the extra corridors in the png image
    l1=30
    l2=15
    # Crop the regions of interest from both images
    cropped_pgm = img_pgm[y-e:y+h+e, x-l1-e:x+w+e]
    cropped_png = img_png[b+l2:b+d, a:a+c-l2]

    # Rotate the pgm image 90 degrees clockwise
    rotated_pgm = np.rot90(cropped_pgm, k=-1)

    # Takes the size of the png image
    target_size = cropped_png.shape
    print("Target size:", target_size)

    # Resize the pmg images to the "target size" (size of the png image)
    resized_pgm = cv2.resize(rotated_pgm, target_size, interpolation=cv2.INTER_NEAREST)
  
    # Return the cropped and resized images as NumPy arrays
    return np.array(resized_pgm), np.array(cropped_png)

    
#save NumPy arrays representing images as PNG files
def save_images_as_png(images, filenames):
    for image, filename in zip(images, filenames):
        # Convert the NumPy array to PIL Image
        pil_image = Image.fromarray(image)

        # Save the PIL Image as PNG
        pil_image.save(filename, "PNG")
        print(f"Saved {filename}")


# Define image paths
pgm_image_path = "Com_gmapping.pgm"
png_image_path = "Com_algoritmo.png"


# call "crop" function
pgm_cropped, png_resized = crop_and_resize_images(pgm_image_path, png_image_path)

# List of NumPy arrays representing images and corresponding filenames (to use save_images_as_png())
images = [pgm_cropped, png_resized]  # List of NumPy arrays representing images
filenames = ["pgm_cropped.png","png_resized.png"]  # List of corresponding filenames

# call "save images as PNG files" function
save_images_as_png(images, filenames)







