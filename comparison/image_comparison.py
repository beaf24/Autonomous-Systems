# import the necessary packages
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
from simpleicp import PointCloud, SimpleICP
from PIL import Image
from icp import icp

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
	imA, imB = np.array(imageA, dtype="float"), np.array(imageB, dtype="float")
	# print(imA-imB.transpose())
	err = np.sum((imA - imB.transpose()) ** 2)
	err = err/float(imageA.shape[0] * imageA.shape[1])

	print("MSE: " + str(err))
	
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
	# print("ok")
	# Create simpleICP object, add point clouds, and run algorithm!
	icp = SimpleICP()
	icp.add_point_clouds(pc_gmapping, pc_occupancy)
	# print("ok")
	H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=1)

	print(X_mov_transformed)
	plt.imshow(H)
	plt.show()

def MSDNN(ground_truth, X, metric:str = "cosine"):
	"""Computes the error with cosine metric"""
	neigh = NearestNeighbors(n_neighbors=1, metric=metric).fit(ground_truth)
	neigh_dist, _ = neigh.kneighbors(X)

	adnn = np.sum(neigh_dist**2)/neigh_dist.shape[0]
	return adnn

def ADNN(ground_truth, X, metric:str = "cosine"):
	"""Computes the error with cosine metric"""
	neigh = NearestNeighbors(n_neighbors=1, metric=metric).fit(ground_truth)
	neigh_dist, _ = neigh.kneighbors(X)

	adnn = np.sum(neigh_dist)/neigh_dist.shape[0]
	return adnn

def get_compare(groud_truth_file: str, image_file:str, resolution:float):
	# Open images and convert to grayscale and array
	groud_truth = np.array(Image.open(groud_truth_file).convert("L"))
	image = np.array(Image.open(image_file).convert("L"))

	# Find positions of interest
	pc_ground_truth = np.argwhere(groud_truth <= 100)*resolution
	pc_image = np.argwhere(image <= 100)*resolution


	# Adjust to resolution
	map_ground_truth = np.int0(pc_ground_truth/resolution)
	map_image = np.int0(pc_image/resolution)

	# Confirm positions of interest
	compare_map = np.zeros((max(map_ground_truth[:,0].max(), map_image[:,0].max())+1, max(map_ground_truth[:,1].max(), map_image[:,1].max())+1))
	compare_map[map_image[:,0] - map_image[:,0].min(), map_image[:,1]- map_image[:,1].min()] = 1
	compare_map[map_ground_truth[:,0]- map_ground_truth[:,0].min(), map_ground_truth[:,1]- map_ground_truth[:,1].min()] = 2

	# Iterative closest point
	transformation_history, new_pc_image = icp(pc_ground_truth, pc_image, distance_threshold= 500, max_iterations=1000000, point_pairs_threshold=2000, verbose=True)
	map_new_image = np.int0(new_pc_image/resolution)

	# Confirm transformation
	compare_map = np.zeros((max(map_image[:,0].max(), map_new_image[:,0].max(), map_ground_truth[:,0].max())+1, max(map_image[:,1].max(), map_new_image[:,1].max(), map_ground_truth[:,1].max())+1))
	compare_map[map_ground_truth[:,0] - map_ground_truth[:,0].min(), map_ground_truth[:,1]- map_ground_truth[:,1].min()] = 1
	# compare_map[map_image[:,0]- map_image[:,0].min(), map_image[:,1]- map_image[:,1].min()] = 2
	compare_map[(map_new_image[:,0]-map_new_image[:,0].min()), (map_new_image[:,1] - map_new_image[:,1].min())] = 3
	plt.imshow(compare_map)
	plt.show()

	# Metrics
	adnn = ADNN(pc_ground_truth, new_pc_image, metric = "euclidean")
	msdnn = MSDNN(pc_ground_truth, new_pc_image, metric = "euclidean")
	print("ADDN: " + str(adnn))
	print("MSDDN: " + str(msdnn))
	print(pc_ground_truth.shape, pc_image.shape)

# # gmapping = np.array(Image.open(os.getcwd() + "/comparison/" + "gmapping_compare.png"))
# gmapping = np.array(Image.open(os.getcwd() + "/comparison/" + "pgm_cropped_dinis.png"))
# occupancy = np.array(Image.open(os.getcwd() + "/comparison/" + "png_resized_dinis.png"))#[:, :, 0]
# mse(gmapping, occupancy)
# # print(np.unique(occupancy))
# pc_gmapping = np.argwhere(gmapping <= 50)
# pc_occupancy = np.argwhere(occupancy <= 50)

# # print(pc_occupancy[:,0] - pc_occupancy[:,0].min())

# ## CONFIRMATION
# confirm_g = np.zeros(gmapping.shape)
# confirm_g[pc_gmapping[:,0], pc_gmapping[:,1]] = 1

# confirm_o = np.zeros(occupancy.shape)
# confirm_o[pc_occupancy[:,0], pc_occupancy[:,1]] = 1

# # compare_map = np.zeros((max(occupancy.shape[0], gmapping.shape[0]), max(occupancy.shape[1], gmapping.shape[1])))
# compare_map = np.zeros((max(pc_occupancy[:,0].max(), pc_gmapping[:,0].max())+1, max(pc_occupancy[:,1].max(), pc_gmapping[:,1].max())+1))
# compare_map[pc_gmapping[:,0] - pc_gmapping[:,0].min(), pc_gmapping[:,1]- pc_gmapping[:,1].min()] = 1
# compare_map[pc_occupancy[:,0]- pc_occupancy[:,0].min(), pc_occupancy[:,1]- pc_occupancy[:,1].min()] = 2

# # plt.imshow(compare_map)
# # plt.show()

# ## CONVERT TO 3D DATA (for ICP)
# # final_gmapping = np.zeros((pc_gmapping.shape[0], 3))
# # final_gmapping[:, 0:2] = pc_gmapping
# # final_occupancy = np.zeros((pc_occupancy.shape[0], 3))
# # final_occupancy[:, 0:2] = pc_occupancy[:, 0:2]
# # print(final_occupancy)
# print(pc_gmapping.shape)
# print(pc_occupancy.shape)
# transformation_history, new_pc_occupancy = icp(pc_gmapping*0.05, pc_occupancy*0.05, verbose=True)
# print(transformation_history, new_pc_occupancy-pc_occupancy)
# # iterative_closest_point(pc_gmapping, pc_occupancy)

# print(np.int0(new_pc_occupancy[:,0]- new_pc_occupancy[:,0].min()))

# compare_map = np.zeros((np.int0(max(pc_occupancy[:,0].max(), new_pc_occupancy[:,0].max()/0.05, pc_gmapping[:,0].max()))+1, np.int0(max(pc_occupancy[:,1].max(), new_pc_occupancy[:,1].max()/0.05, pc_gmapping[:,1].max()))+1))
# compare_map[pc_gmapping[:,0] - pc_gmapping[:,0].min(), pc_gmapping[:,1]- pc_gmapping[:,1].min()] = 1
# compare_map[pc_occupancy[:,0]- pc_occupancy[:,0].min(), pc_occupancy[:,1]- pc_occupancy[:,1].min()] = 2
# compare_map[np.int0((new_pc_occupancy[:,0]-new_pc_occupancy[:,0].min())/0.05), np.int0((new_pc_occupancy[:,1] - new_pc_occupancy[:,1].min())/0.05)] = 3

# plt.imshow(compare_map)
# plt.show()

# adnn = ADNN(pc_gmapping*0.05, new_pc_occupancy, metric = "euclidean")
# msdnn = MSDNN(pc_gmapping*0.05, new_pc_occupancy, metric = "euclidean")
# print("ADDN: " + str(adnn))
# print("MSDDN: " + str(msdnn))

# compare_images(gmapping, occupancy, "title")

if __name__ == "__main__":
	gmapping = "/maps/" + input("Gmapping file: ")
	map = "/maps/" + input("Map to compare: ")
	res = input("resolution: ")
	get_compare(os.getcwd() + gmapping, os.getcwd() + map, resolution=float(res))