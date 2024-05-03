from queue import Full
import cv2
import numpy as np
import streamlit as st
from PIL import Image

    
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
        #uses radius to adjust rectangle to surround edges
        r = dot.size/2
        min_x = min(dot.pt[0]-r, min_x)
        min_y = min(dot.pt[1]-r, min_y)
        max_x = max(dot.pt[0]+r, max_x)
        max_y = max(dot.pt[1]+r, max_y)
    return int(min_x), int(min_y), int(max_x-min_x), int(max_y-min_y)

        

    

    
    
    
    
def create_detector():
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
    params.minArea = 100
    
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


def show_image(img, image_name):
    """shows image in window

    Args:
        img (image): image to be shown
        image_name (String): name of window
    """
    cv2.imshow(image_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_corners(dots):
    """ given array of dots, find x, y, width, and height of the bounding
    rectangle of the braille characters
    Args:
        dots (array of KeyPoints): Array containing information about all braill dots
        
    Returns:
        x, y, w, h: coordinate of top left corner, width, height
    """
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