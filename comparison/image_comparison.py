# import the necessary packages
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
from simpleicp import PointCloud, SimpleICP
from PIL import Image

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

def iterative_closest_point(ground_truth, X):
	"""Serve para obter a matriz de transformação que permite obter o 
	minimo erro entre ground_truth e X
	...
	Ainda não consegui bons resultados com dados 2D
	...
	"""
	np.savetxt("pc_gmapping.xyz", ground_truth)
	np.savetxt("pc_occupancy.xyz", X)

	X_mapping = np.genfromtxt("pc_gmapping.xyz")
	X_occupancy = np.genfromtxt("pc_occupancy.xyz")

	#Create point cloud objects
	pc_gmapping = PointCloud(X_mapping, columns = ["x", "y", "z"])
	pc_occupancy = PointCloud(X_occupancy, columns = ["x", "y", "z"])
	print("ok")
	# Create simpleICP object, add point clouds, and run algorithm!
	icp = SimpleICP()
	icp.add_point_clouds(pc_gmapping, pc_occupancy)
	print("ok")
	H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=1)

	print(X_mov_transformed)
	plt.imshow(H)
	plt.show()

def ADNN(ground_truth, X, metric:str = "cosine"):
	"""Computes the error with cosine metric"""
	neigh = NearestNeighbors(n_neighbors=1, metric=metric).fit(ground_truth)
	neigh_dist, _ = neigh.kneighbors(X)

	adnn = np.sum(neigh_dist**2)/neigh_dist.shape[0]
	return adnn
	

# gmapping = np.array(Image.open(os.getcwd() + "/comparison/" + "gmapping_compare.png"))
gmapping = np.array(Image.open(os.getcwd() + "/comparison/" + "gmapping_compare.png"))
occupancy = np.array(Image.open(os.getcwd() + "/comparison/" + "algorithm_compare.png"))[:, :, 0]
print(np.unique(gmapping))
pc_gmapping = np.argwhere(gmapping == 0)
pc_occupancy = np.argwhere(occupancy == 0)

print(pc_occupancy[:,0] - pc_occupancy[:,0].min())

## CONFIRMATION
confirm_g = np.zeros(gmapping.shape)
confirm_g[pc_gmapping[:,0], pc_gmapping[:,1]] = 1

confirm_o = np.zeros(occupancy.shape)
confirm_o[pc_occupancy[:,0], pc_occupancy[:,1]] = 1

# compare_map = np.zeros((max(occupancy.shape[0], gmapping.shape[0]), max(occupancy.shape[1], gmapping.shape[1])))
compare_map = np.zeros((max(pc_occupancy[:,0].max(), pc_gmapping[:,0].max())+1, max(pc_occupancy[:,1].max(), pc_gmapping[:,1].max())+1))
compare_map[pc_gmapping[:,0] - pc_gmapping[:,0].min(), pc_gmapping[:,1]- pc_gmapping[:,1].min()] = 1
compare_map[pc_occupancy[:,0]- pc_occupancy[:,0].min(), pc_occupancy[:,1]- pc_occupancy[:,1].min()] = 2

plt.imshow(compare_map)
plt.show()

## CONVERT TO 3D DATA (for ITC)
# final_gmapping = np.zeros((pc_gmapping.shape[0], 3))
# final_gmapping[:, 0:2] = pc_gmapping
# final_occupancy = np.zeros((pc_occupancy.shape[0], 3))
# final_occupancy[:, 0:2] = pc_occupancy[:, 0:2]
# print(final_occupancy)

adnn = ADNN(pc_gmapping, pc_occupancy, metric = "euclidean")
print(adnn)

# compare_images(gmapping, occupancy, "title")