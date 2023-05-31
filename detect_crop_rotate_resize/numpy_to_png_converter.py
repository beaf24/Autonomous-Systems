import cv2
import numpy as np
from PIL import Image


#save NumPy arrays representing images as PNG files
def save_images_as_png(images, filenames):
    for image, filename in zip(images, filenames):
        # Convert the NumPy array to PIL Image
        pil_image = Image.fromarray(image)

        # Save the PIL Image as PNG
        pil_image.save(filename, "PNG")
        print(f"Saved {filename}")



# Create a random 2D image of size 100x100
cropped_image1 = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

# Create a gradient image of size 200x200
x = np.linspace(0, 1, 200)
y = np.linspace(0, 1, 200)
X, Y = np.meshgrid(x, y)
cropped_image2 = (255 * (X + Y)).astype(np.uint8)


# List of NumPy arrays representing images and corresponding filenames (to use save_images_as_png())
images = [cropped_image1, cropped_image2]  # List of NumPy arrays representing images
filenames = ["cropped_image1.png","cropped_image2.png"]  # List of corresponding filenames

# call "save images as PNG files" function
save_images_as_png(images, filenames)
