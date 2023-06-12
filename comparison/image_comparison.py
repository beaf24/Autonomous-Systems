# import the necessary packages
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
from simpleicp import PointCloud, SimpleICP
from PIL import Image
from icp import icp
import copy

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
	## Ground truth
	pc_ground_truth = np.argwhere((groud_truth >= 250) | (groud_truth <= 100))*resolution
	pc_ground_truth_free = np.argwhere(groud_truth >= 250)*resolution
	pc_ground_truth_occupied = np.argwhere(groud_truth <= 100)*resolution

	## Algo
	pc_image = np.argwhere((image >= 250) | (image <= 100))*resolution
	pc_image_free = np.argwhere(image >= 250)*resolution
	pc_image_occupied = np.argwhere(image <= 100)*resolution
	
	# Adjust to resolution
	map_ground_truth = np.intp(pc_ground_truth/resolution)
	map_image = np.intp(pc_image/resolution)

	map_ground_truth_free = np.intp(pc_ground_truth_free/resolution)
	map_ground_truth_occupied = np.intp(pc_ground_truth_occupied/resolution)


	# Confirm positions of interest
	# compare_map = np.zeros(((max(map_ground_truth[:,0].max() - map_ground_truth[:,0].min(), map_image[:,0].max())-map_image[:,0].min()) +1, max(map_ground_truth[:,1].max()-map_ground_truth[:,1].min(), map_image[:,1].max()-map_image[:,1].min())+1))
	# compare_map[map_image[:,0] - map_image[:,0].min(), map_image[:,1]- map_image[:,1].min()] = 1
	# compare_map[map_ground_truth[:,0]- map_ground_truth[:,0].min(), map_ground_truth[:,1]- map_ground_truth[:,1].min()] = 2

	## OCCUPIED
	# Iterative closest point
	_, new_pc_image_occupied = icp(pc_ground_truth_occupied, pc_image_occupied, distance_threshold= 100, max_iterations=100, point_pairs_threshold=2000, convergence_rotation_threshold=1e-4, verbose=True)
	map_new_image_occupied = np.intp(new_pc_image_occupied/resolution)

	# Confirm transformation
	compare_map = np.zeros(((max(map_ground_truth_occupied[:,0].max() - map_ground_truth_occupied[:,0].min(), map_new_image_occupied[:,0].max())-map_new_image_occupied[:,0].min()) +1, max(map_ground_truth_occupied[:,1].max()-map_ground_truth_occupied[:,1].min(), map_new_image_occupied[:,1].max()-map_new_image_occupied[:,1].min())+1))
	
	compare_map[map_ground_truth_occupied[:,0] - map_ground_truth_occupied[:,0].min(), map_ground_truth_occupied[:,1]- map_ground_truth_occupied[:,1].min()] = 1
	#compare_map[map_image[:,0]- map_image[:,0].min(), map_image[:,1]- map_image[:,1].min()] = 2
	
	compare_map[(map_new_image_occupied[:,0]-map_new_image_occupied[:,0].min()), (map_new_image_occupied[:,1] - map_new_image_occupied[:,1].min())] = 3
	
	plt.imshow(compare_map)
	plt.show()

	## TOTAL
	# Iterative closest point
	trans2, new_pc_image = icp(pc_ground_truth, pc_image, distance_threshold= 100, max_iterations=200, point_pairs_threshold=2000, verbose=True)
	map_new_image = np.intp(new_pc_image/resolution)

	aligned_free = pc_image_free
	aligned_occupied = pc_image_occupied

	for story in np.arange(len(trans2)):
		rot = trans2[story][0:2, 0:2]
		translation_x = trans2[story][0,2]
		translation_y = trans2[story][1,2]

		aligned_free = np.dot(aligned_free, rot.T)
		aligned_free[:, 0] += translation_x
		aligned_free[:, 1] += translation_y

		aligned_occupied = np.dot(aligned_occupied, rot.T)
		aligned_occupied[:, 0] += translation_x
		aligned_occupied[:, 1] += translation_y

	aligned_free = np.intp(aligned_free/resolution)
	aligned_occupied = np.intp(aligned_occupied/resolution)

	# Confirm transformation
	compare_map = np.zeros(((max(map_ground_truth[:,0].max() - map_ground_truth[:,0].min(), map_new_image[:,0].max()-map_new_image[:,0].min())) +2, max(map_ground_truth[:,1].max()-map_ground_truth[:,1].min(), map_new_image[:,1].max()-map_new_image[:,1].min())+2))
	
	prob_map = np.zeros(((max(aligned_free[:,0].max() - aligned_free[:,0].min(), aligned_occupied[:,0].max()-aligned_occupied[:,0].min())) +2, max(aligned_free[:,1].max()-aligned_free[:,1].min(), aligned_occupied[:,1].max()-aligned_occupied[:,1].min())+2))
	
	gt_class = copy.deepcopy(compare_map)
	gt_class_overall = copy.deepcopy(compare_map)
	algo_class_overall = copy.deepcopy(compare_map)
	algo_class = copy.deepcopy(compare_map)

	
	compare_map[map_ground_truth[:,0] - map_ground_truth[:,0].min(), map_ground_truth[:,1]- map_ground_truth[:,1].min()] = 1
	compare_map[map_ground_truth[:,0] - map_ground_truth[:,0].min()+1, map_ground_truth[:,1]- map_ground_truth[:,1].min()] = 1
	compare_map[map_ground_truth[:,0] - map_ground_truth[:,0].min(), map_ground_truth[:,1]- map_ground_truth[:,1].min()+1] = 1

	compare_map[(map_new_image[:,0]- map_new_image[:,0].min()), (map_new_image[:,1] - map_new_image[:,1].min())] = 3
	compare_map[(map_new_image[:,0]- map_new_image[:,0].min()+1), (map_new_image[:,1] - map_new_image[:,1].min())] = 3
	compare_map[(map_new_image[:,0]- map_new_image[:,0].min()), (map_new_image[:,1] - map_new_image[:,1].min())+1] = 3

	prob_map[(aligned_free[:,0]-aligned_free[:,0].min()), (aligned_free[:,1] - aligned_free[:,1].min())] = 1
	prob_map[(aligned_free[:,0]-aligned_free[:,0].min()+1), (aligned_free[:,1] - aligned_free[:,1].min())] = 1
	prob_map[(aligned_free[:,0]-aligned_free[:,0].min()), (aligned_free[:,1] - aligned_free[:,1].min())+1] = 1

	prob_map[(aligned_occupied[:,0]-aligned_occupied[:,0].min()), (aligned_occupied[:,1] - aligned_occupied[:,1].min())] = 2
	# prob_map[(aligned_occupied[:,0]-aligned_occupied[:,0].min()+1), (aligned_occupied[:,1] - aligned_occupied[:,1].min())] = 2
	# prob_map[(aligned_occupied[:,0]-aligned_occupied[:,0].min()), (aligned_occupied[:,1] - aligned_occupied[:,1].min())+1] = 2

	plt.imshow(prob_map)
	plt.show()

	plt.imshow(compare_map)
	plt.show()

	# Metrics
	adnn_eu = ADNN(pc_ground_truth_occupied, new_pc_image_occupied, metric = "euclidean")
	msdnn_eu = MSDNN(pc_ground_truth_occupied, new_pc_image_occupied, metric = "euclidean")
	adnn_cos = ADNN(pc_ground_truth_occupied, new_pc_image_occupied, metric = "cosine")
	msdnn_cos = MSDNN(pc_ground_truth_occupied, new_pc_image_occupied, metric = "cosine")
	# print("ADDN euclidean: " + str(adnn_eu))
	# print("MSDDN euclidean: " + str(msdnn_eu))
	# print("ADDN cosine: " + str(adnn_cos))
	# print("MSDDN cosine: " + str(msdnn_cos))

	## Classification problem
	# Matrices
	gt_class[map_ground_truth_free[:,0] - map_ground_truth_free[:,0].min(), map_ground_truth_free[:,1]- map_ground_truth_free[:,1].min()] = 1
	gt_class[map_ground_truth_free[:,0] - map_ground_truth_free[:,0].min()+1, map_ground_truth_free[:,1]- map_ground_truth_free[:,1].min()] = 1
	gt_class[map_ground_truth_free[:,0] - map_ground_truth_free[:,0].min(), map_ground_truth_free[:,1]- map_ground_truth_free[:,1].min()+1] = 1

	gt_class[map_ground_truth_occupied[:,0] - map_ground_truth_occupied[:,0].min(), map_ground_truth_occupied[:,1]- map_ground_truth_occupied[:,1].min()] = 2


	algo_class[(aligned_free[:,0]-aligned_free[:,0].min()), (aligned_free[:,1] - aligned_free[:,1].min())] = 1
	algo_class[(aligned_free[:,0]-aligned_free[:,0].min()+1), (aligned_free[:,1] - aligned_free[:,1].min())] = 1
	algo_class[(aligned_free[:,0]-aligned_free[:,0].min()), (aligned_free[:,1] - aligned_free[:,1].min())+1] = 1

	algo_class[(aligned_occupied[:,0]-aligned_occupied[:,0].min()), (aligned_occupied[:,1] - aligned_occupied[:,1].min())] = 2


	gt_class_overall[map_ground_truth[:,0] - map_ground_truth[:,0].min(), map_ground_truth[:,1]- map_ground_truth[:,1].min()] = 1
	gt_class_overall[map_ground_truth[:,0] - map_ground_truth[:,0].min()+1, map_ground_truth[:,1]- map_ground_truth[:,1].min()] = 1
	gt_class_overall[map_ground_truth[:,0] - map_ground_truth[:,0].min(), map_ground_truth[:,1]- map_ground_truth[:,1].min()+1] = 1

	algo_class_overall[(map_new_image[:,0]- map_new_image[:,0].min()), (map_new_image[:,1] - map_new_image[:,1].min())] = 1
	algo_class_overall[(map_new_image[:,0]- map_new_image[:,0].min()+1), (map_new_image[:,1] - map_new_image[:,1].min())] = 1
	algo_class_overall[(map_new_image[:,0]- map_new_image[:,0].min()), (map_new_image[:,1] - map_new_image[:,1].min())+1] = 1

	# Confusion matrix
	cm = confusion_matrix(gt_class.flatten(), algo_class.flatten())
	partial_error = dict()
	metric = ["err_unknown", "err_free", "err_occupied"]
	for i in np.arange(len(cm)):
		partial_error[metric[i]] = 1 - cm[i,i]/cm[i,:].sum()

	print(partial_error)

	error = 1 - accuracy_score(gt_class_overall.flatten(), algo_class_overall.flatten())
	print(cm, error)
	# print(trans1[-1])
	# print(trans2[-1])

	return adnn_eu, msdnn_eu, adnn_cos, msdnn_cos, error, partial_error

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

# print(np.intp(new_pc_occupancy[:,0]- new_pc_occupancy[:,0].min()))

# compare_map = np.zeros((np.intp(max(pc_occupancy[:,0].max(), new_pc_occupancy[:,0].max()/0.05, pc_gmapping[:,0].max()))+1, np.intp(max(pc_occupancy[:,1].max(), new_pc_occupancy[:,1].max()/0.05, pc_gmapping[:,1].max()))+1))
# compare_map[pc_gmapping[:,0] - pc_gmapping[:,0].min(), pc_gmapping[:,1]- pc_gmapping[:,1].min()] = 1
# compare_map[pc_occupancy[:,0]- pc_occupancy[:,0].min(), pc_occupancy[:,1]- pc_occupancy[:,1].min()] = 2
# compare_map[np.intp((new_pc_occupancy[:,0]-new_pc_occupancy[:,0].min())/0.05), np.intp((new_pc_occupancy[:,1] - new_pc_occupancy[:,1].min())/0.05)] = 3

# plt.imshow(compare_map)
# plt.show()

# adnn = ADNN(pc_gmapping*0.05, new_pc_occupancy, metric = "euclidean")
# msdnn = MSDNN(pc_gmapping*0.05, new_pc_occupancy, metric = "euclidean")
# print("ADDN: " + str(adnn))
# print("MSDDN: " + str(msdnn))

# compare_images(gmapping, occupancy, "title")

if __name__ == "__main__":
	name = input("Mapping file: ")
	res_cm = input("Resolution (cm): ")

	gmapping = "/maps/gmapping_" + str(name) + "_" + str(res_cm) + ".png"
	map = "/maps/algo_" + str(name) + "_" + str(res_cm) + ".png"
	adnn_eu, msdnn_eu, adnn_cos, msdnn_cos, error, partials = get_compare(os.getcwd() + gmapping, os.getcwd() + map, resolution=float(1/int(res_cm)))

	# try:
	# 	adnn_eu, msdnn_eu, adnn_cos, msdnn_cos, error = get_compare(os.getcwd() + gmapping, os.getcwd() + map, resolution=float(1/int(res_cm)))
	# except:
	# 	print("Invalid input")
	# 	quit()

	save = input("Y to save: ")
	if save == "Y":
		if os.path.exists("maps/data_compare.txt"):
			file = open("maps/data_compare.txt", "a")
		else:
			file = open("maps/data_compare.txt", "x")
			file.write("Name".ljust(10) + "Resolution".ljust(15) + "ADDN_eu".ljust(15) + "MSDDN_eu".ljust(15) + "ADDN_cos".ljust(15) + "MSDDN_cos".ljust(15) + "Error".ljust(15) + "Error unknown".ljust(15) + "Error free".ljust(15) + "Error occupied" + "\n")
		
		file.write(name.ljust(10) + res_cm.ljust(13) + str(f'{adnn_eu:10f}').ljust(15) + str(f'{msdnn_eu:10f}').ljust(15) + str(f'{adnn_cos:6e}').ljust(15) + str(f'{msdnn_cos:6e}').ljust(15) + str(f'{error:10f}').ljust(15) + str(f'{partials["err_unknown"]:10f}').ljust(15) + str(f'{partials["err_free"]:10f}').ljust(15) + str(f'{partials["err_occupied"]:10f}') + "\n")
		file.close()
	else:
		print("No values were saved, since 'Y' was not pressed.")
	
