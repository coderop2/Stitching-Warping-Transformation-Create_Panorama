### Part 2: Image Transformations

import numpy as np
import math, gc, warnings, scipy, sys
from PIL import Image
import matplotlib.pyplot as plt

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
    final_weights = np.linalg.solve(weights, results)
    matrix = np.array([[final_weights[0], final_weights[1], final_weights[2]],
                      [final_weights[3], final_weights[4], final_weights[5]],
                      [final_weights[6], final_weights[7], 1]])
    #1-(final_weights[6]*source_pts[0][0])-(final_weights[7]*source_pts[0][1])
    if np.allclose(np.dot(weights, final_weights), results):
        return matrix
    else:
        return "Proper solution not found"

if __name__ == '__main__':

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

    print(matrix)

    # Checking if the matrix is singular or not
    if np.linalg.det(matrix) == 0:
        outt = np.zeros(img_asarray.shape)
        output_image = transformations(img_asarray, matrix, outt)
    else:
        output_image = transformations_inverse(img_asarray, matrix)

    output_image.save(output_img_name)


# python3 part2.py part2 1 book2.jpg book1.jpg book_output.jpg 141,131 318,256
# python3 part2.py part2 2 book2.jpg book1.jpg book_output.jpg 141,131 318,256 480,159 534,372
# python3 part2.py part2 3 book2.jpg book1.jpg book_output.jpg 141,131 318,256 480,159 534,372 493,630 316,670
# python3 part2.py part2 4 book2.jpg book1.jpg book_output.jpg 141,131 318,256 480,159 534,372 493,630 316,670 64,601 73,473