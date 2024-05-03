from queue import Full
import cv2
import numpy as np
import streamlit as st
from PIL import Image


"""
Braille Measurements provided by up.codes/s/braille

Dot diameter: 0.059-0.063 inches
Distance between center of dots in the same cell: 0.090-0.100 inches
Distance corresponding dots in adjacent cells: 0.241-0.300 inches
Distance between corresponding dots from one cell directly below: 0.395-0.400 inches
"""
    
def find_bounds(dots):
    """Finds the bounding rectangle for the braille characters

    Args:
        dots (arr[]): list of all detected dots

    Returns:
        ints: x, y, w, h: x and y coordinate of top left corner, width, and height
    """
    
    
    min_x, min_y = dots[0].pt[0], dots[0].pt[1]
    max_y = max_x = 0
    for dot in dots:
        # uses radius to adjust rectangle to surround edges
        r = dot.size/2
        min_x = min(dot.pt[0]-r, min_x)
        min_y = min(dot.pt[1]-r, min_y)
        max_x = max(dot.pt[0]+r, max_x)
        max_y = max(dot.pt[1]+r, max_y)
    # add 1 so when converting to int it rounds up
    return int(min_x), int(min_y), int(max_x-min_x+1), int(max_y-min_y+1)

        


    
    
def crop_to_braille(img, bounding_rect):
    """crops image to only include braille section

    Args:
        img (image): image containing braille
        bounding_rect (x, y, w, h): x and y coords of top left corner, and width and height
    """
    x, y, w, h = bounding_rect
    return img[y:y+h, x:x+w]
    
    
def create_generic_detector():
    """
    Returns:
        blob detector object
    """
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = 1
    # detect darker blobs : 0, detect lighter blobs : 256
    params.blobColor = 0

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 30
    
    # Filter by Circularity
    # 1 = perfect circle, 0.785 is a square
    params.filterByCircularity = True
    params.minCircularity = 0.8
    
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = .1
    
    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = .1

    detector = cv2.SimpleBlobDetector.create(params)
    return detector

def create_detector(img):
    """summary: Uses the size of image to more reliably
    detect braille dots.
    
    
    Args:
        img (image): image of braille characters
    
    Returns:
        blob detector object
    """
    
    # creates detector
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = 0
    # detect darker blobs : 0, detect lighter blobs : 256
    params.blobColor = 0

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 300
    
    # Filter by Area.
    params.filterByArea = True
    img_w, img_h = img.shape[::-1]
    img_area = img_w*img_h
    # Use "find_blob_size" method to determine
    # appropriate minimum area for blobs
    area = find_blob_size(img)
    params.minArea = area
    
    # Filter by Circularity
    # 1 = perfect circle, 0.785 is a square
    params.filterByCircularity = True
    params.minCircularity = 0.5
    
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.1
    
    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.1

    detector = cv2.SimpleBlobDetector.create(params)
    return detector


def find_blob_size(img):
    """Fixes issue with detector detecting dots that are
    too small

    Args:
        img (image): image that is being scanned for dots

    Returns:
        factor: appropriate factor for minArea for a blob
        that prevents small and insignificant blobs from 
        being picked up
    """
    area = img.size//.2
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = area
    params.maxArea = area

    detector = cv2.SimpleBlobDetector.create(params)
    dots = detector.detect(img)
    while len(dots)<2 and area>2:
        area =int(area*0.75)
        params.minArea = int(area)
        detector = cv2.SimpleBlobDetector.create(params)
        dots = detector.detect(img)
    area =int(area*0.85)
    return area



def show_image(img, image_name):
    """shows image in window

    Args:
        img (image): image to be shown
        image_name (String): name of window
    """
    h, w = img.shape[0], img.shape[1]
    factor = 0
    if w>h:
        factor = 500/w
    else:
        factor = 500/h
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, int(factor*w), int(factor*h))
    cv2.imshow(image_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def KeyPoint_clone(dots):
    """Deep Clones dots arra

    Args:
        dots (arr[]): array of dots to be copied
        
    Return:
         (arr[]): deep copy of dots
    """
    return [cv2.KeyPoint(x = k.pt[0], y = k.pt[1], 
            size = k.size, angle = k.angle, 
            response = k.response, octave = k.octave, 
            class_id = k.class_id) for k in dots]
    
def generate_response(dots, img):
    """Accounts for bugged "response" variable in OpenCV's Keypoint class
    Approximates confidence level for each keypoint

    Args:
        dots (arr[]): array of dots
        img (image): images that dots were gathered from
    """
    cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Thresh", 500, 500)
    ret, thresh = cv2.threshold(img, 100,255, cv2.THRESH_BINARY)
    for dot in dots:
        # cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Thresh", 500, 500)
        x, y = dot.pt
        r = dot.size//2
        area = 3.14*r**2
        thresh_copy = thresh[int(y-r):int(y+r+1), int(x-r):int(x+r+1)]
        black_pix = (r*2)**2 - cv2.countNonZero(thresh_copy)
        confidence = black_pix/area
        dot.response = confidence
        # print(confidence)
        # show_image(thresh_copy, "Thresh")
st.set_page_config(page_title="Braille Translator", page_icon="👁️")

st.markdown("# Braille Translator")
st.sidebar.header("Braille Translator")
st.write(
    """This program uses OpenCV to capture and interpret a braille message."""
)

uploaded_file = st.file_uploader("Choose a picture of a puzzle", ['png','jpg'], False)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_cv2 = np.array(image)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    detector = create_detector()


    # Step 1. Identify dots

    dots = detector.detect(img_cv2)

    # draws detected dots
    img_with_keypoints = cv2.drawKeypoints(img_cv2, dots, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



    cv2.imshow('Blob Detection', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





    # Step 2. Create a bounding rectangle


    # Identify size of circles.

    dot_size = dots[0].size

    x,y,w,h = find_bounds(dots)
    print(x)
    print(y)
    print(x+w)
    print(y+h)
    cv2.rectangle(img_cv2, (x,y), (x+w, y+h), (0,255,0),2)

    cv2.imshow('Blob Detection', img_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Step 3. Split rectangle into 6ths
        
        
    # Step 4. Identify which of the six sections are "filled"