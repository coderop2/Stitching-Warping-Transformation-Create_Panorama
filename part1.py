import cv2 
import os
import numpy as np
import sys
from sklearn.cluster import AgglomerativeClustering
## Parse the input arguments ---> argv[1] -- k; argv[:-1] -- images; argv[-1] -- output file; 
## compute the distance matrix
## use this matrix to divide into K clusters
## write this into txt file
## Tunable things  : 1. threshold 2. choice of distance 3. Way clustering is done.
def compute_dist(tup1, tup2):
	keyPoints1 , descriptors1 = tup1
	keyPoints2 , descriptors2 = tup2
	match_count = 0
	threshold = 1.2 ## needs to be tuned
	for i in range(len(keyPoints1)):
		min_dist = float('inf')
		second_min_dist = float('inf')
		for j in range(len(keyPoints2)):
			distance = np.linalg.norm(descriptors1[i] - descriptors2[j])
			##cv2.norm(descriptors1[i], descriptors2[j], cv2.NORM_HAMMING)
			if (distance < min_dist):
				second_min_dist = min_dist
				min_dist = distance
			elif(distance < second_min_dist):
				second_min_dist = distance
		if ((second_min_dist / min_dist) > threshold):
			match_count = match_count + 1
	return match_count


def group_into_clusters(distance_matrix, k):
	clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='complete').fit(distance_matrix)
	return clustering.labels_
def generate_image_groups(output_file_name, image_files_list, num_groups):
	orb = cv2.ORB_create(nfeatures=200)
	feat_dict = {}
	num_images = len(image_files_list)
	dist_mat = np.zeros(shape=(num_images, num_images))
	for i in range(num_images):
		img = cv2.imread(image_files_list[i], cv2.IMREAD_GRAYSCALE)
		if img is None:
			print("File "+image_files_list[i] + " does not exists")
			return
		feat_dict[image_files_list[i]] = orb.detectAndCompute(img, None)
	for i in range(num_images):
		for j in range(i+1, num_images):
			im1 = image_files_list[i]
			im2 = image_files_list[j]
			dist_mat[i][j] = compute_dist(feat_dict[im1], feat_dict[im2])
			dist_mat[j][i] = dist_mat[i][j]
	labels = group_into_clusters(dist_mat, num_groups)
	cluster_to_images = {}
	for i in range(len(labels)):
		if labels[i] not in cluster_to_images:
			cluster_to_images[labels[i]] = []
		cluster_to_images[labels[i]].append(image_files_list[i])
	with open(output_file_name, 'w') as f:
		for _, value in cluster_to_images.items():
			f.write(' '.join(value)+'\n')
if __name__ == "__main__":
	output_file_name = sys.argv[-1]
	num_groups = int(sys.argv[1])
	image_files_list = []
	for i in range(2,len(sys.argv)-1):
		image_files_list.append(sys.argv[i])
	generate_image_groups(output_file_name, image_files_list, num_groups)
	