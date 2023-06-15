# import the necessary packages
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
from simpleicp import PointCloud, SimpleICP
from PIL import Image
from icp import icp, icp_trans
import copy
import os
import argparse
import traceback

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

	#print("MSE: " + str(err))
	
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

	#print(X_mov_transformed)
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

	
	#groud_truth = np.rot90(groud_truth,3)

	print("ground truth size: " + str( groud_truth.size))
	print("image size: " + str( image.size))

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

	new_shape_ground = map_ground_truth.shape
	new_shape_map = pc_image.shape

	
	print(map_ground_truth.shape)
	print(pc_image.shape)

	map_ground_truth_free = np.intp(pc_ground_truth_free/resolution)
	map_ground_truth_occupied = np.intp(pc_ground_truth_occupied/resolution)

	# Confirm positions of interest
	# compare_map = np.zeros(((max(map_ground_truth[:,0].max() - map_ground_truth[:,0].min(), map_image[:,0].max())-map_image[:,0].min()) +1, max(map_ground_truth[:,1].max()-map_ground_truth[:,1].min(), map_image[:,1].max()-map_image[:,1].min())+1))
	# compare_map[map_image[:,0] - map_image[:,0].min(), map_image[:,1]- map_image[:,1].min()] = 1
	# compare_map[map_ground_truth[:,0]- map_ground_truth[:,0].min(), map_ground_truth[:,1]- map_ground_truth[:,1].min()] = 2

	## OCCUPIED
	# Iterative closest point
	trans1, new_pc_image_occupied = icp_trans(pc_ground_truth_occupied, pc_image_occupied, distance_threshold= 50, max_iterations=1000, point_pairs_threshold=50, convergence_translation_threshold=0.001, verbose=True)
	map_new_image_occupied = np.intp(new_pc_image_occupied/resolution)

	# Confirm transformation
	size_x = (max(map_ground_truth_occupied[:,0].max() - map_ground_truth_occupied[:,0].min(), map_new_image_occupied[:,0].max()-map_new_image_occupied[:,0].min()) + np.abs(map_new_image_occupied[:,0].min() - map_ground_truth_occupied[:,0].min())) +1
	size_y = max(map_ground_truth_occupied[:,1].max()-map_ground_truth_occupied[:,1].min(), map_new_image_occupied[:,1].max()-map_new_image_occupied[:,1].min()) + np.abs(map_new_image_occupied[:,1].min() - map_ground_truth_occupied[:,1].min())+1
	compare_map = np.zeros((size_x, size_y))
	
	max_x = min(map_new_image_occupied[:,0].min(), map_ground_truth_occupied[:,0].min()) 
	max_y = min(map_new_image_occupied[:,1].min(), map_ground_truth_occupied[:,1].min())

	compare_map[map_ground_truth_occupied[:,0] - max_x, map_ground_truth_occupied[:,1] - max_y] = 1
	#compare_map[map_image[:,0]- map_image[:,0].min(), map_image[:,1]- map_image[:,1].min()] = 2
	
	compare_map[map_new_image_occupied[:,0] - max_x, map_new_image_occupied[:,1] - max_y] = 3
	# cdict = {'white': [(0.0)], 'blue':[(1.0)], 'red': [(3.0)]}
	plt.imshow(compare_map, cmap="Blues")
	plt.show()

	## TOTAL
	# Iterative closest point
	# trans2, new_pc_image = icp_trans(pc_ground_truth, pc_image, distance_threshold= 800, max_iterations=100, point_pairs_threshold=100, verbose=True)
	
	new_pc_image = pc_image
	aligned_free = pc_image_free
	aligned_occupied = new_pc_image_occupied

	for story in np.arange(len(trans1)):
		translation_x = trans1[story][0]
		translation_y = trans1[story][1]

		print(translation_x, translation_y)
		
		new_pc_image[:, 0] += translation_x
		new_pc_image[:, 1] += translation_y

		aligned_free[:, 0] += translation_x
		aligned_free[:, 1] += translation_y

		# aligned_occupied[:, 0] += translation_x
		# aligned_occupied[:, 1] += translation_y

	aligned_free = np.intp(aligned_free/resolution)
	aligned_occupied = np.intp(aligned_occupied/resolution)
	map_new_image = np.intp(new_pc_image/resolution)

	# Confirm transformation
	size_x_comp = max(map_ground_truth[:,0].max() - map_ground_truth[:,0].min(), map_new_image[:,0].max()-map_new_image[:,0].min()) +np.abs(map_ground_truth[:,0].min() - map_new_image[:,0].min()) +2
	size_y_comp = max(map_ground_truth[:,1].max() - map_ground_truth[:,1].min(), map_new_image[:,1].max()-map_new_image[:,1].min()) + np.abs(map_ground_truth[:,1].min() - map_new_image[:,1].min()) + 2
	compare_map = np.zeros((size_x_comp, size_y_comp))
	
	size_x_prob = max(aligned_free[:,0].max() - aligned_free[:,0].min(), aligned_occupied[:,0].max()-aligned_occupied[:,0].min()) + np.abs(aligned_free[:,0].min() - aligned_occupied[:,0].min()) +2
	size_y_prob = max(aligned_free[:,1].max() - aligned_free[:,1].min(), aligned_occupied[:,1].max()-aligned_occupied[:,1].min()) + np.abs(aligned_free[:,1].min() - aligned_occupied[:,1].min()) + 2
	prob_map = np.zeros((size_x_prob, size_y_prob))
	
	compare_map[map_ground_truth[:,0] - min(map_ground_truth[:,0].min(), map_new_image[:,0].min()), map_ground_truth[:,1]- min(map_ground_truth[:,1].min(), map_new_image[:,1].min())] = 1
	compare_map[map_ground_truth[:,0] - min(map_ground_truth[:,0].min(), map_new_image[:,0].min())+1, map_ground_truth[:,1]- min(map_ground_truth[:,1].min(), map_new_image[:,1].min())] = 1
	compare_map[map_ground_truth[:,0] - min(map_ground_truth[:,0].min(), map_new_image[:,0].min()), map_ground_truth[:,1]- min(map_ground_truth[:,1].min(), map_new_image[:,1].min())+1] = 1

	compare_map[(map_new_image[:,0]- min(map_ground_truth[:,0].min(), map_new_image[:,0].min())), (map_new_image[:,1] - min(map_ground_truth[:,1].min(), map_new_image[:,1].min()))] = 3
	compare_map[(map_new_image[:,0]- min(map_ground_truth[:,0].min(), map_new_image[:,0].min())+1), (map_new_image[:,1] - min(map_ground_truth[:,1].min(), map_new_image[:,1].min()))] = 3
	compare_map[(map_new_image[:,0]- min(map_ground_truth[:,0].min(), map_new_image[:,0].min())), (map_new_image[:,1] - min(map_ground_truth[:,1].min(), map_new_image[:,1].min()))+1] = 3

	prob_map[(aligned_free[:,0]-min(aligned_free[:,0].min(), aligned_occupied[:,0].min())), (aligned_free[:,1] - min(aligned_free[:,1].min(), aligned_occupied[:,1].min()))] = 1
	prob_map[(aligned_free[:,0]-min(aligned_free[:,0].min(), aligned_occupied[:,0].min())+1), (aligned_free[:,1] - min(aligned_free[:,1].min(), aligned_occupied[:,1].min()))] = 1
	prob_map[(aligned_free[:,0]-min(aligned_free[:,0].min(), aligned_occupied[:,0].min())), (aligned_free[:,1] - min(aligned_free[:,1].min(), aligned_occupied[:,1].min()))+1] = 1

	prob_map[(aligned_occupied[:,0]-min(aligned_free[:,0].min(), aligned_occupied[:,0].min())), (aligned_occupied[:,1] - min(aligned_free[:,1].min(), aligned_occupied[:,1].min()))] = 2
	# prob_map[(aligned_occupied[:,0]-aligned_occupied[:,0].min()+1), (aligned_occupied[:,1] - aligned_occupied[:,1].min())] = 2
	# prob_map[(aligned_occupied[:,0]-aligned_occupied[:,0].min()), (aligned_occupied[:,1] - aligned_occupied[:,1].min())+1] = 2

	plt.imshow(prob_map, cmap="Blues")
	plt.show()

	plt.imshow(compare_map, cmap="Blues")
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
	global_x_size = max(size_x_prob, size_x_comp)
	global_y_size = max(size_y_prob, size_y_comp)
	
	global_map = np.zeros((global_x_size, global_y_size))
	
	gt_class = copy.deepcopy(global_map)
	gt_class_overall = copy.deepcopy(global_map)
	algo_class_overall = copy.deepcopy(global_map)
	algo_class = copy.deepcopy(global_map)


	min_x_gt_class = min(map_ground_truth_free[:,0].min(), map_ground_truth_occupied[:,0].min())
	min_y_gt_class = min(map_ground_truth_free[:,1].min(), map_ground_truth_occupied[:,1].min())

	min_x_algo_class = min(aligned_free[:,0].min(), aligned_occupied[:,0].min())
	min_y_algo_class = min(aligned_free[:,1].min(), aligned_occupied[:,1].min())

	global_x_min = min(min_x_algo_class, min_x_gt_class)
	global_y_min = min(min_y_algo_class, min_y_gt_class)

	gt_class[map_ground_truth_free[:,0] - global_x_min, map_ground_truth_free[:,1]- global_y_min] = 1
	gt_class[map_ground_truth_free[:,0] - global_x_min+1, map_ground_truth_free[:,1]- global_y_min] = 1
	gt_class[map_ground_truth_free[:,0] - global_x_min, map_ground_truth_free[:,1]- global_y_min+1] = 1
	gt_class[map_ground_truth_occupied[:,0] - global_x_min, map_ground_truth_occupied[:,1]- global_y_min] = 2

	
	algo_class[(aligned_free[:,0]-global_x_min), (aligned_free[:,1] - global_y_min)] = 1
	algo_class[(aligned_free[:,0]-global_x_min+1), (aligned_free[:,1] - global_y_min)] = 1
	algo_class[(aligned_free[:,0]-global_x_min), (aligned_free[:,1] - global_y_min)+1] = 1
	algo_class[(aligned_occupied[:,0]-global_x_min), (aligned_occupied[:,1] - global_y_min)] = 2

	plt.imshow(gt_class, cmap="Blues")
	plt.show()

	plt.imshow(algo_class, cmap="Blues")
	plt.show()

	overall_x_min = min(map_ground_truth[:,0].min(), map_new_image[:,0].min())
	overall_y_min = min(map_ground_truth[:,1].min(), map_new_image[:,1].min())

	gt_class_overall[map_ground_truth[:,0] - overall_x_min, map_ground_truth[:,1]- overall_y_min] = 1
	gt_class_overall[map_ground_truth[:,0] - overall_x_min+1, map_ground_truth[:,1]- overall_y_min] = 1
	gt_class_overall[map_ground_truth[:,0] - overall_x_min, map_ground_truth[:,1]- overall_y_min+1] = 1
	
	algo_class_overall[(map_new_image[:,0]- overall_x_min), (map_new_image[:,1] - overall_y_min)] = 1
	algo_class_overall[(map_new_image[:,0]- overall_x_min+1), (map_new_image[:,1] - overall_y_min)] = 1
	algo_class_overall[(map_new_image[:,0]- overall_x_min), (map_new_image[:,1] - overall_y_min)+1] = 1

	plt.imshow(gt_class_overall, cmap="Blues")
	plt.show()
	
	plt.imshow(algo_class_overall, cmap="Blues")
	plt.show()
	# Confusion matrix
	cm = confusion_matrix(gt_class.flatten(), algo_class.flatten())
	partial_error = dict()
	metric = ["err_unknown", "err_free", "err_occupied"]
	for i in np.arange(len(cm)):
		partial_error[metric[i]] = 1 - cm[i,i]/cm[i,:].sum()

	#print(partial_error)

	error = 1 - accuracy_score(gt_class_overall.flatten(), algo_class_overall.flatten())
	#print(cm, error)
	# print(trans1[-1])
	# print(trans2[-1])

	return adnn_eu, msdnn_eu, adnn_cos, msdnn_cos, error, partial_error

if __name__ == "__main__":
	#name = input("Mapping file: ")
	#res_cm = input("Resolution (cm): ")

	parent_dir = os.getcwd()

	
	parser = argparse.ArgumentParser()
	parser.add_argument("-p_free", "--pfree", type=str)
	parser.add_argument("-res", "--res", type=str)
	parser.add_argument("-name", "--name", type=str)
	args = parser.parse_args()

	if args.pfree == "all":
		pfrees = np.round(np.arange(0.10, 0.50, 0.01),2)
		for args.pfree in pfrees:
			print(args.pfree)
			gmapping = parent_dir + "/automated-with-comparison/mapping-comparison/images/sr manel - pfree/map.png"
			map = parent_dir + "/automated-with-comparison/mapping-comparison/images/sr manel - pfree/map_algo-" + str(args.pfree) + ".png"

			print(map)
			
			adnn_eu, msdnn_eu, adnn_cos, msdnn_cos, error, partials = get_compare(gmapping, map, resolution=0.1)
			
			

			#save = input("Y to save: ")
			save = "Y"
			if save == "Y":
				if os.path.exists(parent_dir + "/automated-with-comparison/mapping-comparison/data/data_compare.txt"):
					file = open(parent_dir + "/automated-with-comparison/mapping-comparison/data/data_compare.txt", "a")
				else:
					file = open(parent_dir + "/automated-with-comparison/mapping-comparison/data/data_compare.txt", "x")
					file.write("Name\tResolution\tLog free\tADDN_eu\tMSDDN_eu\tADDN_cos\tMSDDN_cos\tError\tError unknown\tError free\tError occupied\n")
							
				file.write(str(args.name) + "\t" + str(args.res) + "\t" + str(args.pfree) + "\t" + str(adnn_eu) + "\t" + str(msdnn_eu) + "\t" + str(adnn_cos) + "\t" + str(msdnn_cos) + "\t" + str(error) + "\t" + str(partials["err_unknown"]) + "\t" + str(partials["err_free"]) + "\t" + str(partials["err_occupied"]) + "\n")
				file.close()
			else:
				print("No values were saved, since 'Y' was not pressed.")
	else:
		gmapping = parent_dir + "/automated-with-comparison/mapping-comparison/images/sr manel - pfree/map.png"
		map = parent_dir + "/automated-with-comparison/mapping-comparison/images/sr manel - pfree/map_algo-" + str(args.pfree) + ".png"

		print(map)
		
		adnn_eu, msdnn_eu, adnn_cos, msdnn_cos, error, partials = get_compare(gmapping, map, resolution=0.1)
		
		

		#save = input("Y to save: ")
		save = "Y"
		if save == "Y":
			if os.path.exists(parent_dir + "/automated-with-comparison/mapping-comparison/data/data_compare.txt"):
				file = open(parent_dir + "/automated-with-comparison/mapping-comparison/data/data_compare.txt", "a")
			else:
				file = open(parent_dir + "/automated-with-comparison/mapping-comparison/data/data_compare.txt", "x")
				file.write("Name\tResolution\tLog free\tADDN_eu\tMSDDN_eu\tADDN_cos\tMSDDN_cos\tError\tError unknown\tError free\tError occupied\n")
						
			file.write(str(args.name) + "\t" + str(args.res) + "\t" + str(args.pfree) + "\t" + str(adnn_eu) + "\t" + str(msdnn_eu) + "\t" + str(adnn_cos) + "\t" + str(msdnn_cos) + "\t" + str(error) + "\t" + str(partials["err_unknown"]) + "\t" + str(partials["err_free"]) + "\t" + str(partials["err_occupied"]) + "\n")
			file.close()
		else:
			print("No values were saved, since 'Y' was not pressed.")

