#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import sys
import copy
import pdb
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def get_slope(x1,y1,x2,y2):
    return 1.0*(y2-y1)/(x2-x1)

def draw_lines(img, lines, color=[255, 0, 0], thickness=6, output_path="test_images/annotated/"):
    """workflow:
    1) examine each individual line returned by hough & determine if it's in left or right lane by its slope
    because we are working "upside down" with the array, the left lane will have a negative slope and right positive
    2) track extrema
    3) compute averages
    4) solve for b intercept 
    5) use extrema to solve for points
    6) smooth frames and cache
    """
    y_global_min = img.shape[0] #min will be the "highest" y value, or point down the road away from car
    y_max = img.shape[0]
    l_slope, r_slope = [],[]
    l_lane,r_lane = [],[]
    det_slope = 0.4
    hough_lines_img = copy.deepcopy(img)

    print "Found {0} lines in the image.".format(len(lines))
    for line in lines:
        #1
        for x1,y1,x2,y2 in line:
            cv2.line(hough_lines_img, (x1, y1), (x2, y2), color, thickness)
            slope = get_slope(x1,y1,x2,y2)
            if slope > det_slope:
                r_slope.append(slope)
                r_lane.append(line)
            elif slope < -det_slope:
                l_slope.append(slope)
                l_lane.append(line)
        #2
        y_global_min = min(y1,y2,y_global_min)

    mpimg.imsave(output_path + "6a-hough_lines.jpg", hough_lines_img)
    
    
    # to prevent errors in challenge video from dividing by zero
    if((len(l_lane) == 0) or (len(r_lane) == 0)):
        print ('no lane detected')
        return 1
        
    #3
    l_slope_mean = np.mean(l_slope,axis =0)
    r_slope_mean = np.mean(r_slope,axis =0)
    l_mean = np.mean(np.array(l_lane),axis=0)
    r_mean = np.mean(np.array(r_lane),axis=0)
    
    if ((r_slope_mean == 0) or (l_slope_mean == 0 )):
        print('dividing by zero')
        return 1
        
    #4, y=mx+b -> b = y -mx
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])
    
    #5, using y-extrema (#2), b intercept (#4), and slope (#3) solve for x using y=mx+b
    # x = (y-b)/m
    # these 4 points are our two lines that we will pass to the draw function
    l_x1 = int((y_global_min - l_b)/l_slope_mean) 
    l_x2 = int((y_max - l_b)/l_slope_mean)   
    r_x1 = int((y_global_min - r_b)/r_slope_mean)
    r_x2 = int((y_max - r_b)/r_slope_mean)
    
    #6
    if l_x1 > r_x1:
        l_x1 = int((l_x1+r_x1)/2)
        r_x1 = l_x1
        l_y1 = int((l_slope_mean * l_x1 ) + l_b)
        r_y1 = int((r_slope_mean * r_x1 ) + r_b)
        l_y2 = int((l_slope_mean * l_x2 ) + l_b)
        r_y2 = int((r_slope_mean * r_x2 ) + r_b)
    else:
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max
      
    vertices = np.array([l_x1,l_y1,l_x2,l_y2,r_x1,r_y1,r_x2,r_y2],dtype=np.int32)
    cv2.line(img, (vertices[0], vertices[1]), (vertices[2], vertices[3]), color, thickness)
    cv2.line(img, (vertices[4], vertices[5]), (vertices[6], vertices[7]), color, thickness)
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, output_path="test_images/annotated/"):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, output_path=output_path)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)

def process_frame(image, output_path="test_images/annotated/"):

    imshape = image.shape
    print "imshape =", imshape

    # KITCHEN
    # lower_left = [0, imshape[0]]
    # lower_right = [imshape[1]*9/10, imshape[0]]
    # top_left = [0, imshape[0]/4]
    # top_right = [imshape[1]*7/8, imshape[0]/4]

    # HIGHWAY
    # lower_left = [0, imshape[0]]
    # lower_right = [imshape[1]*7/9, imshape[0]]
    # top_left = [imshape[1]*3/8, imshape[0]*4/10]
    # top_right = [imshape[1]*6/9, imshape[0]*4/10]

    # BALCANI
    # lower_left = [imshape[1]/8, imshape[0]]
    # lower_right = [imshape[1]*7/8, imshape[0]]
    # top_left = [imshape[1]*3/7, imshape[0]*5/8]
    # top_right = [imshape[1]*4/7, imshape[0]*5/8]

    # DEFAULT
    lower_left = [imshape[1]/9, imshape[0]]
    lower_right = [imshape[1]*8/9, imshape[0]]
    top_left = [imshape[1]*3/8, imshape[0]*6/10]
    top_right = [imshape[1]*5/8, imshape[0]*6/10]

    roi_lines_image = copy.deepcopy(image)
    color = [0, 255, 0]
    thickness = 3
    cv2.line(roi_lines_image, (lower_left[0], lower_left[1]), (top_left[0], top_left[1]), color, thickness)
    cv2.line(roi_lines_image, (top_left[0], top_left[1]), (top_right[0], top_right[1]), color, thickness)
    cv2.line(roi_lines_image, (top_right[0], top_right[1]), (lower_right[0], lower_right[1]), color, thickness)
    cv2.line(roi_lines_image, (lower_right[0], lower_right[1]), (lower_left[0], lower_left[1]), color, thickness)
    mpimg.imsave(output_path + "1a-roi_lines_image.jpg", roi_lines_image)

    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    roi_image = region_of_interest(image, vertices)
    mpimg.imsave(output_path + "1b-roi_image.jpg", roi_image)

    gray_image = grayscale(roi_image)
    mpimg.imsave(output_path + "1-grayscale.jpg", gray_image)

    img_hsv = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
    mpimg.imsave(output_path + "2-hsv.jpg", img_hsv)
    #hsv = [hue, saturation, value]
    #more accurate range for yellow since it is not strictly black, white, r, g, or b

    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mpimg.imsave(output_path + "3a-mask_yelow.jpg", mask_yellow)

    mask_white = cv2.inRange(gray_image, 180, 255)
    mpimg.imsave(output_path + "3b-mask_white.jpg", mask_white)

    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)
    mpimg.imsave(output_path + "3c-mask_yellow_white.jpg", mask_yw_image)

    kernel_size = 5
    gauss_gray = gaussian_blur(mask_yw_image,kernel_size)
    mpimg.imsave(output_path + "4-gauss_gray.jpg", gauss_gray)

    #same as quiz values
    low_threshold = 50
    high_threshold = 150
    canny_edges = canny(gauss_gray,low_threshold,high_threshold)
    mpimg.imsave(output_path + "5-canny_edges.jpg", canny_edges)

    #rho and theta are the distance and angular resolution of the grid in Hough space
    #same values as quiz
    rho = 2
    theta = np.pi/180
    #threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 30
    min_line_len = 50
    max_line_gap = 100

    line_image = hough_lines(canny_edges, rho, theta, threshold, min_line_len, max_line_gap, output_path=output_path)
    mpimg.imsave(output_path + "6b-hough_lines.jpg", line_image)

    result = weighted_img(line_image, image, alpha=0.8, beta=1., gamma=0.)
    return result

def main():
    if not os.path.exists("test_images/annotated"):
        os.mkdir("test_images/annotated", 0755)
    for source_img in os.listdir("test_images/"):
        if source_img.endswith(".jpg") and source_img[:-4] == "YellowUnderShade2":
            output_path = "test_images/annotated/" + source_img[:-4] + "/"
            if not os.path.exists(output_path):
                os.mkdir(output_path, 0755)

            image = mpimg.imread("test_images/" + source_img)
            print "Image = ", source_img
            processed = process_frame(image, output_path=output_path)
            mpimg.imsave(output_path + "8-annotated_" + source_img, processed)


if __name__ == "__main__":
    main()