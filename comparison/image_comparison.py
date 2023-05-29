# import the necessary packages
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from simpleicp import PointCloud, SimpleICP
from PIL import Image
import icp


## 1.
# Aplicar o algoritmo Iterative closest point (ICP) para obter a matriz de transformação que permite obter o
# erro minimo entre o mapa obtido e a groud truth
## 2. 
# Calcular a average distance to the nearest neighbour (ADNN)
# DOI: 10.1109/ICARCV.2018.8581131

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
	# show the images
	plt.show()
	
# load the images -- the original, the original + contrast,
# and the original + photoshop

gmapping = np.array(Image.open(os.getcwd() + "/comparison/" + "gmapping_compare.png"))
occupancy = np.array(Image.open(os.getcwd() + "/comparison/" + "algorithm_compare.png"))
print(np.unique(occupancy))
pc_gmapping = np.argwhere(gmapping == 254)
pc_occupancy = np.argwhere(occupancy == 255)
print(pc_occupancy[:, 0:2])
# transformation, points = icp.icp(pc_gmapping, pc_occupancy)
np.savetxt("pc_gmapping.xy", pc_gmapping)
np.savetxt("pc_occupancy.xy", pc_occupancy)

X_mapping = np.genfromtxt("pc_gmapping.xy")
X_occupancy = np.genfromtxt("pc_occupancy.xy")

#Create point cloud objects
pc_gmapping = PointCloud(X_mapping, columns = ["x", "y"])
pc_occupancy = PointCloud(X_occupancy, columns = ["x", "y"])
print("ok")
# Create simpleICP object, add point clouds, and run algorithm!
icp = SimpleICP()
icp.add_point_clouds(pc_gmapping, pc_occupancy)
print("ok")
H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=1)

# compare_images(gmapping, occupancy, "title")