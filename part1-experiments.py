import cv2 
import os
import numpy as np
import sys
from sklearn.cluster import AgglomerativeClustering
import random
## Parse the input arguments ---> argv[1] -- k; argv[:-1] -- images; argv[-1] -- output file; 
## compute the distance matrix
## use this matrix to divide into K clusters
## write this into txt file
## Tunable things  : 1. threshold 2. choice of distance 3. Way clustering is done.
def compute_dist(tup1, tup2, threshold):
	keyPoints1 , descriptors1 = tup1
	keyPoints2 , descriptors2 = tup2
	match_count = 0
	##threshold = 1.2 ## needs to be tuned
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


def group_into_clusters(distance_matrix, k, link):
	clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage=link).fit(distance_matrix)
	return clustering.labels_
def generate_image_groups(output_file_name, image_files_list, num_groups, num, link, threshold):
	orb = cv2.ORB_create(nfeatures=num)
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
			dist_mat[i][j] = compute_dist(feat_dict[im1], feat_dict[im2], threshold)
			dist_mat[j][i] = dist_mat[i][j]
	labels = group_into_clusters(dist_mat, num_groups, link)
	cluster_to_images = {}
	for i in range(len(labels)):
		if labels[i] not in cluster_to_images:
			cluster_to_images[labels[i]] = []
		cluster_to_images[labels[i]].append(image_files_list[i])
	with open(output_file_name, 'w') as f:
		for _, value in cluster_to_images.items():
			f.write(' '.join(value)+'\n')
def get_tp_count(l):
	tp = 0
	n = len(l)
	for i in range(n):
		for j in range(i+1,n):
			if (l[i].split('_')[0] == l[j].split('_')[0]):
				tp = tp + 1
	return tp
def get_tn_count(l1, l2):
	tn = 0
	n1 = len(l1)
	n2 = len(l2)
	for i in range(n1):
		for j in range(n2):
			if (l1[i].split('_')[0] != l2[j].split('_')[0]):
				tn = tn + 1
	return tn
def get_metrics(label_dict, index, lst):
	values = []
	n = 0
	for _, value in label_dict.items():
		n = n + len(value)
		values.append(value)
	k = len(values)
	tp = 0
	tn = 0
	for i in range(k):
		tp = tp + get_tp_count(values[i])
		for j in range(i+1, k):
			tn = tn + get_tn_count(values[i], values[j])
	f = open("measurements_"+str(index)+".txt", 'w')
	f.write('Features:'+str(lst[0])+'\n')
	f.write('Links:'+lst[1]+'\n')
	f.write('thresholds'+str(lst[2])+'\n')
	f.write('TP:'+str(tp)+'\n')
	f.write('TN:'+str(tn)+'\n')
	f.write('PairWise Accuracy:'+str((tp+tn)*2 / (n*(n-1))))
	f.close()
if __name__ == "__main__":
	output_file_name = sys.argv[-1]
	num_groups = int(sys.argv[1])
	image_files_list = []
	for i in range(2,len(sys.argv)-1):
		image_files_list.append(sys.argv[i])
	thresholds = [1.1, 1.2, 1.5, 1.8]
	links = ['average','complete', 'single']
	num_features = [20, 50,  100,  150, 200, 300]
	all_list = [(f, l, t) for f in num_features for l in links for t in thresholds]
	params_list = random.sample(all_list, 1)
	for i in range(len(params_list)):
		generate_image_groups(output_file_name+"_"+str(i)+".txt", image_files_list, num_groups, params_list[i][0], params_list[i][1], params_list[i][2])
		label_dict = {}
		j = 0
		with open(output_file_name+"_"+str(i)+".txt",'r') as f:
			lines = [line.rstrip() for line in f]
			for line in lines:
				label_dict[j] = line.split()
				j = j + 1
		get_metrics(label_dict, i, params_list[i])
	