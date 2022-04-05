import numpy as np
import math, gc, warnings, scipy, sys, cv2
from PIL import Image
import matplotlib.pyplot as plt

def projective(source_pts, dest_pts):
    # print(source_pts)
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
    return matrix

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
    
if __name__=="__main__":
    # read the images in grayscale using cv2 library 
    # The reason for chossing grayscale is that the algo can focus on the 
    # features of the image can not the intensities in diifferent channels of the image
    img1 = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    output_img = sys.argv[3]
    
    # I have done best feature selecion in 3 different methods, these are:
    # METHOD 1: Choosing 1 best for each 1000 points in the image 1 that matches
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

    img_1 = cv2.imread(sys.argv[1])
    img_2 = cv2.imread(sys.argv[2])
    new_img = np.zeros((new_y+10, new_x+10, 3))

    try:
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

        cv2.imwrite(sys.argv[3], temp)
    except:
        cv2.imwrite(sys.argv[3], img_2)
    # print(hypothesis)
    # print(matching_points)
    # print(len(matching_points))