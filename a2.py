from subprocess import call
import cv2 
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import math, gc, warnings, scipy, sys
from PIL import Image


############################
# TO-DO #
# need to make changes for the command line argument for each part
# Add try and catch blocks
############################

############################
# Benefits #
# Will give as added layer of protection against potential bugs
# Will give modularity
############################

####Functions for part 1###
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
            
#####End of part1#####

####Functions for part 2#####
def bilinear_interploation(x,y,image):
    img_height,img_width=image.shape[:2]
    x1=math.floor(x)
    y1=math.floor(y)
    x2=x1+1
    y2=y1+1
    a=x-x1
    b=y-y1
    W=(1-a)*(1-b)*image[x1][y1][0]
    W1=(1-a)*(1-b)*image[x1][y1][1]
    W2=(1-a)*(1-b)*image[x1][y1][2]
    if x2>=img_height:
        Y=0
        Y1=0
        Y2=0
    else:
        Y=a*(1-b)*image[x2][y1][0]
        Y1=a*(1-b)*image[x2][y1][1]
        Y2=a*(1-b)*image[x2][y1][2]

    if x2>=img_height or y2>=img_width:
        Z=0
        Z1=0
        Z2=0
    else:
        Z=a*b*image[x2][y2][0]
        Z1=a*b*image[x2][y2][1]
        Z2=a*b*image[x2][y2][2]

    if y2>=img_width:
        X=0
        X1=0
        X2=0
    else:
        X=(1-a)*b*image[x1][y2][0]
        X1=(1-a)*b*image[x1][y2][1]
        X2=(1-a)*b*image[x1][y2][2]

    val=W+X+Y+Z
    val1=W1+X1+Y1+Z1
    val2=W2+X2+Y2+Z2
    return val,val1,val2

def transformations_inverse(img_asarray, transformation_matrix):
    inverse_transformation_matrix=np.linalg.inv(transformation_matrix)
    img_height,img_width,img_depth=img_asarray.shape[:3]
    # print(img_height,img_width,img_depth)
    img_new=np.zeros((img_height,img_width,img_depth))
    # print(img_new.shape)


    for row in range(img_height):
        for col in range(img_width):
            a=np.array([col,row,1])
            a=a.reshape(3,1)
            co_ord=np.dot(inverse_transformation_matrix,a)
            # print(co_ord)
            y=co_ord[0][0]/co_ord[2][0]
            x=co_ord[1][0]/co_ord[2][0]

            if x>=0 and x<img_height and y>=0 and y<img_width:
                val,val1,val2=bilinear_interploation(x,y,img_asarray)
                img_new[row][col][0]=val
                img_new[row][col][1]=val1
                img_new[row][col][2]=val2

    return Image.fromarray(img_new.astype('uint8'))

def transformations(input_image, tranformation_matrix, output_image):
    inp_img = input_image.copy()
    out_img = output_image.copy()
    trans = tranformation_matrix
    temp_out = np.zeros((out_img.shape[0], out_img.shape[1]))

    inp_x = inp_img.shape[1]
    inp_y = inp_img.shape[0]

    for x in range(inp_x):
        for y in range(inp_y):
            curr_cord = np.array([x, y, 1])
            out = np.dot(tranformation_matrix, curr_cord)
            # print(out)
            out_x = int(out[0]/out[-1])
            out_y = int(out[1]/out[-1])
            # print(out_x, out_y)

            if (out_x > 0 and out_x < inp_x) and (out_y > 0 and out_y < inp_y) and temp_out[out_y, out_x] != 1:
                out_img[out_y, out_x, :] = inp_img[y, x, :]
                temp_out[out_y, out_x] = 1


    # print("Missing pixels are numbered", 2092032 - np.sum(temp_out))
    # if np.sum(temp_out) != out_img.shape[0]*out_img.shape[1]:
    #     pass # Then do bileanr interpolation
    return Image.fromarray(out_img.astype('uint8'))

def translation(source_pt, dest_pt):
    matrix = np.array([[1, 0, dest_pt[0] - source_pt[0]],
                      [0, 1, dest_pt[1] - source_pt[1]],
                      [0, 0, 1]])
    return matrix


def euclidean_rigid(source_pts, dest_pts):
    weights = np.array([[source_pts[0][0], -source_pts[0][1], 1, 0],
                        [source_pts[0][1], source_pts[0][0], 0, 1],
                        [source_pts[1][0], -source_pts[1][1], 1, 0],
                        [source_pts[1][1], source_pts[1][0], 0, 1]])
    results = np.array([dest_pts[0][0],
                        dest_pts[0][1],
                        dest_pts[1][0],
                        dest_pts[1][1]])
    final_weights = np.linalg.solve(weights, results)
    matrix = np.array([[final_weights[0], -final_weights[1], final_weights[2]],
                      [final_weights[1], final_weights[0], final_weights[3]],
                      [0, 0, 1]])
    if np.allclose(np.dot(weights, final_weights), results):
        return matrix
    else:
        return "Proper solution not found"

def affine(source_pts, dest_pts):
    weights = np.array([[source_pts[0][0], source_pts[0][1], 1, 0, 0, 0],
                        [0, 0, 0, source_pts[0][0], source_pts[0][1], 1],
                        [source_pts[1][0], source_pts[1][1], 1, 0, 0, 0],
                        [0, 0, 0, source_pts[1][0], source_pts[1][1], 1],
                        [source_pts[2][0], source_pts[2][1], 1, 0, 0, 0],
                        [0, 0, 0, source_pts[2][0], source_pts[2][1], 1]])
    results = np.array([dest_pts[0][0],
                        dest_pts[0][1],
                        dest_pts[1][0],
                        dest_pts[1][1],
                        dest_pts[2][0],
                        dest_pts[2][1]])
    final_weights = np.linalg.solve(weights, results)
    matrix = np.array([[final_weights[0], final_weights[1], final_weights[2]],
                      [final_weights[3], final_weights[4], final_weights[5]],
                      [0, 0, 1]])
    if np.allclose(np.dot(weights, final_weights), results):
        return matrix
    else:
        return "Proper solution not found"

def projective(source_pts, dest_pts):
    weights = np.array([[source_pts[0][0], source_pts[0][1], 1, 0, 0, 0,-source_pts[0][0]*dest_pts[0][0],-source_pts[0][1]*dest_pts[0][0]],
                        [0, 0, 0, source_pts[0][0], source_pts[0][1], 1,-source_pts[0][0]*dest_pts[0][1],-source_pts[0][1]*dest_pts[0][1]],
                        [source_pts[1][0], source_pts[1][1], 1, 0, 0, 0,-source_pts[1][0]*dest_pts[1][0],-source_pts[1][1]*dest_pts[1][0]],
                        [0, 0, 0, source_pts[1][0], source_pts[1][1], 1,-source_pts[1][0]*dest_pts[1][1],-source_pts[1][1]*dest_pts[1][1]],
                        [source_pts[2][0], source_pts[2][1], 1, 0, 0, 0,-source_pts[2][0]*dest_pts[2][0],-source_pts[2][1]*dest_pts[2][0]],
                        [0, 0, 0, source_pts[2][0], source_pts[2][1], 1,-source_pts[2][0]*dest_pts[2][1],-source_pts[2][1]*dest_pts[2][1]],
                        [source_pts[3][0], source_pts[3][1], 1, 0, 0, 0,-source_pts[3][0]*dest_pts[3][0],-source_pts[3][1]*dest_pts[3][0]],
                        [0, 0, 0, source_pts[3][0], source_pts[3][1], 1,-source_pts[3][0]*dest_pts[3][1],-source_pts[3][1]*dest_pts[3][1]]])
    results = np.array([dest_pts[0][0],
                        dest_pts[0][1],
                        dest_pts[1][0],
                        dest_pts[1][1],
                        dest_pts[2][0],
                        dest_pts[2][1],
                        dest_pts[3][0],
                        dest_pts[3][1]])
    # Check for singularity
    if np.linalg.det(weights) == 0:
        return None
    final_weights = np.linalg.solve(weights, results)
    matrix = np.array([[final_weights[0], final_weights[1], final_weights[2]],
                      [final_weights[3], final_weights[4], final_weights[5]],
                      [final_weights[6], final_weights[7], 1]])
    #1-(final_weights[6]*source_pts[0][0])-(final_weights[7]*source_pts[0][1])
    if np.allclose(np.dot(weights, final_weights), results):
        return matrix
    else:
        return "Proper solution not found"
    
#### Part 2 ends#####

#### Functions for part3####
def find_descriptors(img1, img2, n = 1000):
    orb = cv2.ORB_create(nfeatures = n)
    (keypoints1, descriptors1) = orb.detectAndCompute(img1, None)
    (keypoints2, descriptors2) = orb.detectAndCompute(img2, None)
    return (keypoints1,descriptors1),(keypoints2,descriptors2)

def find_distance(img1_data, img2_data, num_points, threshold = 100, use_cv = False):
    keypoints1,descriptors1=img1_data
    keypoints2,descriptors2=img2_data
    all_dists = []
    matching_points=[]
    matching_points_thresh = []
    cv_list = []
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches_ = sorted(matches, key = lambda x:x.distance, reverse = True)
    for ele in matches_:
        cv_list.append(((img1_data[0][ele.queryIdx].pt, img1_data[1][ele.queryIdx]), 
                 (img2_data[0][ele.trainIdx].pt, img2_data[1][ele.trainIdx])))
        
    for i in range(len(keypoints1)):
        distances=dict()
        k=0
        for j in range(len(keypoints2)):
            dist = cv2.norm(descriptors1[i], descriptors2[j], cv2.NORM_HAMMING)
            # dist = np.linalg.norm(descriptors1[i] - descriptors2[j])
            distances[keypoints2[j]] = dist
            all_dists.append(dist)
            if dist < threshold:
                matching_points_thresh.append((dist, ((keypoints1[i].pt, descriptors1[i]),
                                               (keypoints2[j].pt, descriptors2[j]))))

        distances=sorted(distances.items(), key= lambda item:item[1], reverse=False)
        for idx,(key,value) in enumerate(distances):
            if k<num_points:
                matching_points.append(((keypoints1[i].pt, descriptors1[i]),
                                        (key.pt, descriptors2[idx])))
                k=k+1
            else:
                break
    matching_points_thresh = sorted(matching_points_thresh, key= lambda x:x[0], reverse=False)[:100]
    matching_points_thresh = [item[1] for item in matching_points_thresh]
    return matching_points, matching_points_thresh, cv_list, all_dists

def get_src_dest(matching_points_temp, index_pts):
    src_pts = []
    dest_pts = []
    for x in np.random.choice(index_pts, 4, False):
        src_pts.append(matching_points_temp[x][0])
        dest_pts.append(matching_points_temp[x][1])
        index_pts.remove(x)
    return src_pts, dest_pts, index_pts

def RANSAC(pts, threshold = 1):
    hypothesis = []
    matching_pts = []
    for x in pts:
        matching_pts.append(((int(x[0][0][0]), int(x[0][0][1])), (int(x[1][0][0]), int(x[1][0][1]))))
    for i in range(100): # hyperparameter of RANSAC which needs to be tuned
        index_pts = list(range(len(matching_pts)))
        src_pts, dest_pts, index_pts = get_src_dest(matching_pts, index_pts)
        hypo = projective(dest_pts, src_pts)
        # print(np.sum(hypo))
        if hypo is None:
            continue
        votes = 0
        
        while len(index_pts) > 3:
            src_pts, dest_pts, index_pts = get_src_dest(matching_pts, index_pts)
            curr_hypo = projective(dest_pts, src_pts)
            if curr_hypo is None:
                continue
            if abs(np.sum(hypo) - np.sum(curr_hypo)) < threshold:
                votes += 1
        if votes > 0:
            # print(list(hypo))
            hypothesis.append((votes, hypo)) #[list(hypo)] = votes
        
    return hypothesis

def RANSAC_naive(matching_pts, threshold = 1):
    hypothesis = {}
    for i in range(500): # hyperparameter of RANSAC which needs to be tuned
        matching_points_temp = matching_points.copy()
        index_pts = list(range(len(matching_points_temp)))
        idx = np.random.choice(index_pts)
        hypo = matching_points_temp[idx]
        hypo_calc = np.sum(abs(hypo[0][1] - hypo[1][1]))
        # matching_points_temp.pop(idx)
        index_pts.remove(idx)
        votes = 0
        while len(index_pts) > 0:
            idx = np.random.choice(index_pts)
            curr_hypo = matching_points_temp[idx]
            curr_hypo_calc = np.sum(abs(curr_hypo[0][1] - curr_hypo[1][1]))
            # matching_points_temp.pop(idx)
            index_pts.remove(idx)
            if hypo_calc - curr_hypo_calc < threshold:
                vote += 1
        if vote > 0:
            hypothesis[hypo] = votes
        
    return hypothesis

def part3(img1_name, img2_name, output_img_name):
    # read the images in grayscale using cv2 library 
    # The reason for chossing grayscale is that the algo can focus on the 
    # features of the image can not the intensities in diifferent channels of the image
    img1 = cv2.imread(img1_name, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_name, cv2.IMREAD_GRAYSCALE)
    output_img = output_img_name

    # I have done best feature selecion in 3 different methods, these are:
    # METHOD 1: Choosing 1 best for each 10000 points in the image 1 that matches
    # a point in image 2
    # METHOD 2: Use a threshold to choose which point combination are there between 2 
    # images which are passed as input
    # METHOD 3: Use the input CV BFMatcher function to get the best matches between
    # the points which were found using ORB
    n_points = 2
    img1_data,img2_data = find_descriptors(img1,img2)
    matching_points, matching_points_thresh, cv_list, all_dists = find_distance(img1_data,img2_data,n_points)

    # Run RANSAC for n (here we are running it for 100 iterations which can be
    # adjusted according to needs and requirements - as we are chosing the 4 points
    # by random so every iteration we will get different poinst from the last
    # iterations). Different RANSAC method empolyed are:
    # Naive RANSAC : Which consideres each and every pair matched in the previous
    # step as a hypothesis 
    # Refined RANSAC : Here we develop a hypothesis by selecting 4 matched pairs to 
    # form a projective matrix and then to the voting to form inliers and outliers
    hypothesis = RANSAC(matching_points_thresh)
    # hypothesis = RANSAC(matching_points)
    # hypothesis = RANSAC_naive(matching_points_thresh)
    matrix = sorted(hypothesis, key = lambda x: x[0])[-1][1]

    # Get the corners of the image that we are going to transform in this case 
    # it is the second image to get new corners which are going to be the 
    # corners of the image after applying tranformation
    corners = np.array([[0, 0, 1], 
         [img2.shape[1], 0, 1], 
         [0, img2.shape[0], 1], 
         [img2.shape[1], img2.shape[0], 1]])
    new_corners = np.dot(matrix, corners.T)

    # Next part is about getting the 
    min_x = min(new_corners[0])
    min_y = min(new_corners[1])
    max_x = max(new_corners[0])
    max_y = max(new_corners[1])
    if min_x < 0:
        new_x = int(max(max_x + abs(min_x), img1.shape[1] + abs(min_x)))
        shift_x = abs(min_x)
    else:
        new_x = int(max(max_x, img1.shape[1]))
        shift_x = 0
    if min_y < 0:
        new_y = int(max(max_y + abs(min_y), img1.shape[0] + abs(min_y)))
        shift_y = abs(min_y)
    else:
        new_y = int(max(max_y, img1.shape[0]))
        shift_y = 0

    img_1 = cv2.imread(img1_name)
    img_2 = cv2.imread(img2_name)
    
    try:
        new_img = np.zeros((new_y+10, new_x+10, 3))

        for x in range(img_2.shape[1]):
            for y in range(img_2.shape[0]):
                out = np.dot(matrix, np.array([x, y, 1]).T)
                new_img[int(out[1]//out[-1] + shift_y), int(out[0]//out[-1] + shift_x), :] = img_2[y, x, :]

        temp = new_img.copy()
        for x in range(img_1.shape[1]):
            for y in range(img_1.shape[0]):
                if temp[y, x, 0] != 0:
                    temp[int(y + shift_y), int(x + shift_x), :] = (temp[int(y + shift_y), int(x + shift_x), :] + img_1[y, x, :])/2
                else:
                    temp[int(y + shift_y), int(x + shift_x), :] = img_1[y, x, :]

        cv2.imwrite(output_img, temp)
        print('Panorama Created')
    except:
        cv2.imwrite(output_img, img_2)
    # print(hypothesis)
    # print(matching_points)
    # print(len(matching_points))

####end of part 3####

if __name__=="__main__":
    
    if sys.argv[1]=='part1':
        #3 bigben_2.jpg bigben_7.jpg bigben_8.jpg eiffel_18.jpg eiffel_19.jpg colosseum_8.jpg colosseum_11.jpg clustering2.txt
        output_file_name = sys.argv[-1]
        num_groups = int(sys.argv[2])
        image_files_list = []
        for i in range(3,len(sys.argv)-1):
            image_files_list.append(sys.argv[i])
        generate_image_groups(output_file_name, image_files_list, num_groups)
        
    elif sys.argv[1]=='part2':
        n=int(sys.argv[2])

        output_img_name = sys.argv[5]

        im=Image.open(sys.argv[3])
        img_asarray=np.asarray(im)

        matrix=np.array([[0.907,0.258,-182],
                         [-0.153,1.44,58],
                         [-0.000306,0.000731,1]])

        if n==1:
            if len(sys.argv)<8:
                print("Please provide proper arguments for Translation transformation")
            else:
                source = eval(sys.argv[6])
                dest = eval(sys.argv[7])
                matrix = translation(source,dest)

        elif n==2:
            if len(sys.argv)<10:
                print("Please provide proper arguments for Rigid transformation")
            else:
                source = [eval(sys.argv[6]), eval(sys.argv[8])]
                dest = [eval(sys.argv[7]), eval(sys.argv[9])]
                matrix = euclidean_rigid(source,dest)

        elif n==3:
            if len(sys.argv)<12:
                print("Please provide proper arguments for Affine transformation")
            else:
                source = [eval(sys.argv[6]), eval(sys.argv[8]), eval(sys.argv[10])]
                dest = [eval(sys.argv[7]), eval(sys.argv[9]), eval(sys.argv[11])]
                matrix = affine(source,dest)

        elif n==4:
            if len(sys.argv)<14:
                print("Please provide proper arguments for Projective transformation")
            else:
                source = [eval(sys.argv[6]), eval(sys.argv[8]), eval(sys.argv[10]), eval(sys.argv[12])]
                dest = [eval(sys.argv[7]), eval(sys.argv[9]), eval(sys.argv[11]), eval(sys.argv[13])]
                matrix = projective(source,dest)

        # print(matrix)

        # Checking if the matrix is singular or not
        if np.linalg.det(matrix) == 0:
            outt = np.zeros(img_asarray.shape)
            output_image = transformations(img_asarray, matrix, outt)
        else:
            output_image = transformations_inverse(img_asarray, matrix)

        output_image.save(output_img_name)
    	#call(["python3", "part2.py", "test-images/image-1.jpg", "outputs/inputs"])
        
    elif sys.argv[1]=='part3':
    	part3(sys.argv[2], sys.argv[3], sys.argv[4])