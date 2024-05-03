import cv2
import imutils
import pytesseract
import numpy as np
import transform
from skimage.segmentation import clear_border
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import streamlit as st
from PIL import Image
# from doctr.models import ocr_predictor


# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# def ocr(item):
#     model = ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True)
#     result = model(item)
#     json_output = result.export()
#     return result, json_output

def orderPoints(pts):
    """
    Organizes list of points into top right, top left, bottom right, and bottom left
    param pts: list of coordinate
    :return: ordered list of points : (tl, tr, br, bl)
    """
    rect = np.zeros((4, 2), dtype = "float32")
    #top left will have the lowest sum between x and y value
    #bottom right will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    #top right will have the smallest (including negaive) difference between x and y
    #bottom left will have the greatest difference between x and y
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect
def fourPointTransform(image, pts):
    """
    Corrects the perspective of a rectangle to be perfectly rectangular
    param image: image]
    param pts: list of 4 points of the rectangle
    :return: image
    """
    rect = orderPoints(pts)
    (tl, tr, br, bl) = rect
    
    #compute the new image, which will be the maximum distance between either the top two corners, or bottom two corners
    widthA = np.sqrt(((br[0]-bl[0]) ** 2) + ((br[1]-bl[1]) ** 2))
    widthB = np.sqrt(((tr[0]-tl[0]) ** 2) + ((tr[1]-tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    #compute height of the new image, which will be the maximum distance between left two corners, or the right two corners
    heightA = np.sqrt(((tr[0]-br[0]) ** 2) + ((tr[1]-br[1]) ** 2))
    heightB = np.sqrt(((tl[0]-bl[0]) ** 2) + ((tl[1]-bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    #creates array with the dimensions of the new image
    dms = np.array([
        [0,0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype= "float32")

    #calculates the transformation matrix to feed into cv2.warpPerspective
    M = cv2.getPerspectiveTransform(rect, dms)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    #return warped image
    return warped

def findPuzzle(img):
    """
    Finds the border of the puzzle
    :param img: Image of board
    :return: Bounding Rectangle of puzzle
    """

    contours = []
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (9, 9), 3)
 



    #apply threshhold
    thresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #invert thresh
    thresh = cv2.bitwise_not(thresh)
    

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    #initialize puzzle contour
    puzzleCnt = None
    #finds the largest square shaped contour
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03*perimeter, True)

        if len(approx) == 4:
            puzzleCnt = approx
            break
    thresh = fourPointTransform(thresh, puzzleCnt.reshape(4,2))
    img = fourPointTransform(img, puzzleCnt.reshape(4,2))
    
    #isolates the puzzle from the rest of the image
    cv2.imshow('thresh',thresh)
    if cv2.waitKey(0) & 0xff == ord('q'):
        cv2.destroyAllWindows()
    return thresh, img


def findCells(thresh):
    """
    Splits puzzle into individual cells
    Uses extractNumber() to identify the number in the cell, then puts that info in a table
    :param thresh: cropped-image
    :return: 9x9 list of integers
    """
    puzzTable = np.zeros((9, 9), dtype="int")

    #iterate through each cell and fill in array at the same time
    #create is blank function that uses thresh
    Xstep = thresh.shape[0] // 9
    Ystep = thresh.shape[1] // 9
    for x in range(0,9):
        startX = x * Xstep
        endX = (x + 1) * Xstep
        for y in range(0,9):
            startY = y * Ystep
            endY = (y + 1) * Ystep
            cell = thresh[startX: endX, startY: endY]
            puzzTable[x,y] = extractNumber(cell)

    return puzzTable


def cropCell(cell):
    """
    Crops the outer 1/10th of the cell in order to prevent
    the program from mistaking the borders as part of the number.
    It crops more off the sides, because numbers tend to be taller than
    they are wide, and the right and left sides of the box are the most
    likely to cause problems
    param cell: binary outline of cell
    :return: cropped cell
    """
    Xsize = cell.shape[0]
    Ysize = cell.shape[1]
    Xtrim = Xsize // 10
    Ytrim = Ysize // 20
    return cell[Xtrim: Xsize - Xtrim, Ytrim: Ysize - Ytrim]


def extractNumber(thresh):
    """
    param thresh: binary outline of cell
    :return: integer
    """
    
    
    thresh = clear_border(thresh)
    # cv2.imshow('thresh',thresh)
    # if cv2.waitKey(0) & 0xff == ord('q'):
    #     cv2.destroyAllWindows()
    thresh = cv2.resize(thresh,(28,28))
    thresh = thresh.reshape((1,28,28,1))
    #text = new_model.predict(thresh)
    text, temp = ocr(thresh)
    text_digit = text.argmax()
    print(text)
    #handling any blank cells
    if len(text) == 0:
        return 0
    if text[0] == '':
        return 0
    num = int(text[0])
    if num > 9:
        if(num/10 == 1):
            return num%10
        elif(num%10 == 1):
            return num/10
        else:
            return 0
    print(num)
    return num

def solve(bd):
    """
    Solves a sudoku board using backtracking
    :param bd: 2d list of ints
    :return: solution
    """
    empty_cell = empty(bd)
    if empty_cell:
        row, col = empty_cell
    else:
        return True
    for i in range(1, 10):
        if valid(bd, (row, col), i):
            bd[row][col] = i

            if (solve(bd)):
                return True

            bd[row][col] = 0

    return False


def valid(bd, pos, num):
    """
    Checks if the move is valid
    :param bo: 2d list of ints
    :param pos: position on board (row,col)
    :param num: attempted input
    :return: True or False
    """
    # Check row
    for i in range(0, len(bd)):
        if bd[pos[0]][i] == num and pos[1] != i:
            return False
    # Check col
    for i in range(0, len(bd)):
        if bd[i][pos[1]] == num and pos[0] != i:
            return False
    # Check box

    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if bd[i, j] == num and (i, j) != pos:
                return False
    return True

def mapEmpty(bd):
    empty_cells = set()
    for i in range(len(bd)):
        for j in range(len(bd[i])):
            if bd[i][j] == 0:
                empty_cells.add((i,j))
                
    return empty_cells
def empty(bd):
    """
    finds empty cell in board
    :param bo: 2d list of ints
    :return: (int, int) row, col
    """
    for i in range(len(bd)):
        for j in range(len(bd[i])):
            if bd[i][j] == 0:
                return (i, j)

    return None

def putOnImage(table, img, empty_cells):
    """
    Fills in the picture of the sudoku board
    param: table: a 9x9 2d array that is the solved puzzle
    param: img: the image of the puzzle
    :return: img
    
    """
    
    Xstep = img.shape[0] // 9
    Ystep = img.shape[1] // 9
    for x in range(0,9):
        startX = x * Xstep
        endX = (x + 1) * Xstep
        for y in range(0,9):
            startY = y * Ystep
            endY = (y + 1) * Ystep
            loc = [(x * Xstep + Xstep // 2), (y * Ystep + Xstep // 2)]
            if (y,x) in empty_cells:
                cv2.putText(img, str(table[y][x]), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 4)
                cv2.imshow('Solution',img)
                if cv2.waitKey(0) & 0xff == ord('q'):
                    cv2.destroyAllWindows()
    return img

st.set_page_config(page_title="Sudoku Solver", page_icon="🧩")

st.markdown("# Sudoku Solver")
st.sidebar.header("Sudoku Solver")
st.write(
    """This program uses OpenCV and Tensoflow OCR to recognize and parse information from a picture
    of a sudoku puzzle, then runs a fairly simple backtracking algorithm to solve it."""
)

uploaded_file = st.file_uploader("Choose a picture of a puzzle", ['png','jpg'], False)


# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     img_cv2 = np.array(image)
#     img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2BGRA)
#     board, img = findPuzzle(img_cv2)
#     table = findCells(board)
#     empty_cells = mapEmpty(table)
#     print(table)
#     print("\n")
#     solve(table)
#     print(table)
#     img = putOnImage(table, img, empty_cells)
#     img = cv2.resize(img, (1280,960))
#     cv2.imshow('Solution',img)
#     if cv2.waitKey(0) & 0xff == ord('q'):
#         cv2.destroyAllWindows()











