import cv2
import numpy as np
from PIL import Image
import streamlit as st

# st.set_page_config(page_title="Braille Translator", page_icon="👁️")

Braille_to_Letters = {
    "100000":"a",
    "110000":"b",
    "100100":"c",
    "100110":"d",
    "100010":"e",
    "110100":"f",
    "110110":"g",
    "110010":"h",
    "010100":"i",
    "010110":"j",
    "101000":"k",
    "111000":"l",
    "101100":"m",
    "101110":"n",
    "101010":"o",
    "111100":"p",
    "111110":"q",
    "111010":"r",
    "011100":"s",
    "011110":"t",
    "101001":"u",
    "111001":"v",
    "010111":"w",
    "101101":"x",
    "101111":"y",
    "101011":"z",
}

Braille_to_Numbers = {
    "100000":"1",
    "110000":"2",
    "100100":"3",
    "100110":"4",
    "100010":"5",
    "110100":"6",
    "110110":"7",
    "110010":"8",
    "010100":"9",
    "010110":"0",
}

upper_case_sign = "001000"
number_sign = "001111"
letter_sign = "000011"


    
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
    # cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Thresh", 500, 500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for dot in dots:
        # cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Thresh", 500, 500)
        x, y = dot.pt
        r = dot.size//2
        area = 3.14*(r)**2
        gray_cropped = thresh[int(y-r):int(y+r+1), int(x-r):int(x+r+1)]
        black_pix = (r*2)**2 - cv2.countNonZero(gray_cropped)
        confidence = black_pix/area
        dot.response = confidence
        # print(confidence)
        # show_image(thresh_copy, "Thresh")
        
def filter_confidence(dots, threshold):
    """Removes dots that are incorrectly idenfified

    Args:
        dots (KeyPoints[]): An array of dots stored as KeyPoints
        threshold (double): A percentage of confidence that a keypoint must be to remain valid
        
    Return:
        filtered_dots (KeyPoints[]): An array of dots stored as KeyPoints with any dot
        whos confidence level is under the given threshold removed
    """
    filtered_dots = list()
    for dot in dots:
        if dot.response>=threshold:
            filtered_dots.append(dot)
    return filtered_dots
    
        
def group_dots(dots, dot_size):
    """find and groups dots that coorespond to the same letter

    Args:
        dots (arr[]): array of dots sorted by x coordinates
        dot_size (int): diamerter of first dot
        
    Return:
        grouped_dots (List(List())): a list filled with lists of dots grouped by x coordinate
    """
    # gap of 1.5 between horizontally
    grouped_dots = []
    grouped_dots.append([dots[0]])
    idx = 0
    for i in range(1, len(dots)):
        if abs(dots[i].pt[0] - dots[i-1].pt[0])<(2*dot_size):
            grouped_dots[idx].append(dots[i])
        else:
            idx += 1
            grouped_dots.append([dots[i]])
    return grouped_dots
    
    
def find_row(dot, x, y, w, h):
    """_summary_

    Args:
        dot (keypoint): Keypoint of braille character
        x (int): left bound for braille characters
        y (int): top bound for braille characters
        w (int): width of braille section
        h (int): height of braille section
        
    Return:
        0 if in top row
        1 if in middle row
        2 if in bottom row
    """
    # dots 1 and 4 will be at any height < y+(1.5*dot_size)
    # dots 2 and 5 will be at any height y+(1.5*dot_size) <= and < y+(3.0*dotsize)
    # dots 3 and 6 will be at any height <= y+(3.0*dot_size)
    
    if dot.pt[1] < y+(1.5*dot.size):
        return 0
    elif dot.pt[1] < y+(3.0*dot.size):
        return 1
    else:
        return 2
    
    
def organize_cell(grouped_dots, x, y, w, h):
    """Organizes every array of dots in grouped dots, into a 
    6 index long array in which each index cooresponds to a braille
    cells dot positions
    
    eg...
    
    1  4
    2  5
    3  6
    
    Args:
        grouped_dots (List(List())): List of groups of dots that belong to a letter
        x (int): left bound for braille characters
        y (int): top bound for braille characters
        w (int): width of braille section
        h (int): height of braille section
        
    Return:
        organized_cell (List(List())): 6xN list of 1s and 0s
    """
    # split 1, 2, and 3 from 4, 5, and 6
    cell_positions = np.zeros((len(grouped_dots), 6), dtype=int)
    idx = 0
    for dots in grouped_dots:
        base = dots[0]
        left = []
        right = []
        for dot in dots:
            # if in the same column as the leftmost dot
            if dot.pt[0] - base.pt[0] < 0.5 * base.size:
                left.append(dot)
            else:
                right.append(dot)
        
        for dot in left:
            cell_positions[idx][find_row(dot, x, y, w, h)] = 1
        
        for dot in right:
            cell_positions[idx][3+find_row(dot, x, y, w, h)] = 1
        idx += 1
    return cell_positions
    

def get_cell_coords(organized_cell):
    """Converts the 6 index array into a 6 character long string

    Args:
        organized_cell (List(List())): 6xN list of 1s and 0s
        
    Return:
        cell_coords (List(List())): the same list as organized cell but as strings
    """    
    cell_coords = []
    for dots in organized_cell:
        str1 = ""
        for b in dots:
            str1 += str(b)
        cell_coords.append(str1)
    return cell_coords

def cell_to_English(organized_cell):
    """Converts 6 index array of 1s and 0s to the alphabet representation

    Args:
        organized_cell (List(List())): 6xN list of 1s and 0s
        
    Return:
        letters (List): List of characters
    """
    letters = list()
    isNumber = False
    cell_coords = get_cell_coords(organized_cell)
    for cell in cell_coords:
        if cell == number_sign:
            isNumber = True
            letters.append(" ")
        elif cell == letter_sign:
            isNumber = False
            letters.append(" ")
        else:
            if isNumber:
                if cell in Braille_to_Numbers.keys():
                    letters.append(Braille_to_Numbers[cell])
                else: 
                    letters.append(" ")
            else:
                if cell in Braille_to_Letters.keys():
                    letters.append(Braille_to_Letters[cell])
                else: 
                    letters.append(" ")
    return letters

def toString(arr):
    s = ""
    for c in arr:
        s = s+str(c)
    return s
            
        
        



st.markdown("# Braille Translator")
st.sidebar.header("Braille Translator")
st.write(
    """This program uses OpenCV to capture and interpret a braille message."""
)

dot_color = st.toggle("White Dots")


uploaded_file = st.file_uploader("Choose a picture of a puzzle", ['png','jpg'], False)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)




    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 1)
    # Create Binary image
    ret, thresh = cv2.threshold(blur, 200,255, cv2.THRESH_BINARY)
    if dot_color:
        thresh = cv2.bitwise_not(thresh)
    # show_image(thresh, "img")

    detector = create_detector(thresh)

    # Step 1. Identify dots

    dots = detector.detect(gray)

    # draws detected dots


    img_with_keypoints = cv2.drawKeypoints(img, dots, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    x,y,w,h = find_bounds(dots)
    cropped = crop_to_braille(img_with_keypoints, (x, y, w, h))
    rectangle_bound = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2, cv2.FONT_HERSHEY_SIMPLEX)
    st.image(rectangle_bound)
    st.image(cropped)
    #show_image(cropped, "cropped")



    # Step 2. Create a bounding rectangle


    # Identify size of circles.
    try:
        dot_size = dots[0].size
    except:
        raise Exception("No dots found...")
        
    x,y,w,h = find_bounds(dots)
    # cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)


    cropped = crop_to_braille(img, (x, y, w, h))

    # sort dots by confidence
    generate_response(dots, img)
    
    # filter out low confidence levels
    dots = filter_confidence(dots, 0.5)
    dots_confidence = dots + ()
    dots_x = dots + ()
    dots_y = dots + ()
    dots_confidence = sorted(dots_confidence, key=lambda KeyPoint: KeyPoint.response, reverse=True)
    # sorts dots based on x value
    dots_x = sorted(dots_x, key=lambda KeyPoint: KeyPoint.pt[0])
    # sorts dots based on y value
    dots_y = sorted(dots_y, key=lambda KeyPoint: KeyPoint.pt[1])
    grouped_dots = group_dots(dots_x, dot_size)
    # put into 2x3 matrix
    cell_coords = organize_cell(grouped_dots, x, y, w, h)

    letters = cell_to_English(cell_coords)

    st.text(toString(letters))
        
        
    # Step 4. Identify which of the six sections are "filled"
