import cv2 as cv
import numpy as np
import pandas as pd
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import chi2_kernel


cap = cv.VideoCapture(0)
curr_time = 0
prev_time = 0

patterns = {
    'H': [
    [[1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1]],

    [[1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1]],

    [[1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1]],

    [[1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1]]

    ]
    ,
    'S': [
    [[0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0]] ,

    [[0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0, 0]],

    [[1, 1, 0, 0, 0, 0, 1, 1, 0],
    [1, 0, 0, 0, 0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 1, 1, 1, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 0, 0, 1]],

    [[0, 0, 0, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0, 1, 1]]
    
    ],

    'U':[
    [[0, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1]],

    [[0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1]],

    [[0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1]],

    [[1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0]],

    [[1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 0, 0]],

    [[1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0]],

    [[1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 0]],

    [[0, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1]]
    ],

    
}

def cosine_matrix_similarity(mat1, mat2):
    flat1 = mat1.flatten().reshape(1, -1)
    flat2 = mat2.flatten().reshape(1, -1)
    return cosine_similarity(flat1, flat2)[0][0]


def ColorCheck(contour):
    x , y , w , h = cv.boundingRect(contour)
    ratio= w/h
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)
    min_cnt = None #set with qualityyyy
    max_cnt = None #sameeee
    return (
        min_cnt < cv.countourArea(contour) <max_cnt and 
        0.25 < ratio < 2.5 and
        3 < len(approx) < 7 
    )

def CheckParts(blob_image, color_ranges, color):
    height, width = blob_image.shape[:2]
    blob_image_hsv = cv.cvtColor(blob_image, cv.COLOR_BGR2HSV)

    block_width = width // 4
    block_height = height // 4

    thresh = None
    for c in color_ranges:
        if c["name"] == color:
            thresh = c
            break
    

    count = 0
    for i in range(4):
        for j in range(4):
            x1 = j * block_width
            x2 = (j + 1) * block_width
            y1 = i * block_height
            y2 = (i + 1) * block_height
            
            block = blob_image_hsv[y1:y2, x1:x2]
                
            h_mean = np.mean(block[:,:,0])
            s_mean = np.mean(block[:,:,1])
            v_mean = np.mean(block[:,:,2])

            if (thresh['lower'][0] <= h_mean <= thresh['upper'][0] and
                thresh['lower'][1] <= s_mean <= thresh['upper'][1] and
                thresh['lower'][2] <= v_mean <= thresh['upper'][2]):
                count += 1
    
    return count >= 14


aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
def ColorDetection(frame):
    cropped = None
    color_found = False

    color_ranges = [
        {
            "name": "Green",
            "lower": np.array([32,134,18]),
            "upper": np.array([140, 255, 108]),
            "display_color": (0, 255, 0)  
        },
        {
            "name": "Red",
            "lower": np.array([0, 114, 140]),
            "upper": np.array([179, 255, 255]),
            "display_color": (0, 0, 255) 
        },
        {
            "name": "Yellow",
            "lower": np.array([20, 100, 100]),
            "upper": np.array([30, 255, 255]),
            "display_color": (0, 255, 255)  
        }
    ]

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    for color in color_ranges:

        mask = cv.inRange(hsv, color["lower"], color["upper"])
        

        kernel = np.ones((5, 5), np.uint8)
        mask = cv.erode(mask, kernel, iterations=1)
        mask = cv.dilate(mask, kernel, iterations=2)
        

        contours, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        

        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            if cv.contourArea(largest_contour) > 500:  
                
                rect = cv.minAreaRect(largest_contour)
                box = cv.boxPoints(rect)
                box = np.int32(box)
                
                width, height = int(rect[1][0]), int(rect[1][1])
                src_pts = box.astype("float32")

                dst_pts = np.array([[0, height-1],
                                    [0, 0],
                                    [width-1, 0],
                                    [width-1, height-1]], dtype="float32")
                
                M = cv.getPerspectiveTransform(src_pts, dst_pts)
                

                cropped = cv.warpPerspective(frame, M, (width, height))
                
                if CheckParts(cropped, color_ranges, color["name"]):
                    color_found = True

                    cv.drawContours(frame, [box], 0, color["display_color"], 2)
                    center = (int(rect[0][0]), int(rect[0][1]))
                    cv.circle(frame, center, 5, color["display_color"], -1)

    return frame, cropped, color

def CheckContours(cnt, frame):
    max_area = 9000
    min_area = 100
    height, width = frame.shape[:2]
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int32(box)
    
    w, h = int(rect[1][0]), int(rect[1][1])
    area = cv.contourArea(cnt)
    x , y , _ , _ = cv.boundingRect(cnt)
    aspect_ratio = w/h if h > 0 else 0

    # roi = frame[y:y+h, x:x+w]
    # mask = np.zeros((h, w), dtype=np.uint8)
    # cnt_offset = cnt - [x, y]  
    # cv.drawContours(mask, [cnt_offset], -1, 255, -1)
    
    # mean_intensity = cv.mean(roi, mask=mask)[0]
    # intensity_ratio = mean_intensity / 255
    # print(aspect_ratio)
    return (
        min_area <= area <= max_area and
        width//16 <= w <= width//2.2 and
        0.55 <= aspect_ratio <= 1.8 and
        x >= 2 and
        x + w <= width - 2 and
        y >= 8 and
        h + y <= height 
    )

def is_black_white_marker(image, corners):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        x, y, w, h = cv.boundingRect(corners)
        margin = 10
        x_min = max(0, x - margin)
        y_min = max(0, y - margin)
        x_max = min(image.shape[1], x + w + margin)
        y_max = min(image.shape[0], y + h + margin)
        background = gray[y_min:y_max, x_min:x_max]
        border = gray[y:y+h, x:x+w]
        mean_intensity = np.mean(background)
        border_intensity = np.mean(border[:5, :])  
        return mean_intensity > 200 and border_intensity < 50


def filters(frame):
    gray = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray , (15 ,15), 1)
    adaptive_mean = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 53, 20)
    invert = (255 - adaptive_mean)

    return invert
def CropVictims(frame):
    cropped = None
    box = None

    gray = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray , (15 ,15), 1)
    adaptive_mean = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 201, 16)
    invert = (255 - adaptive_mean)

    contours , _ = cv.findContours(invert.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours :
        for contour in contours:
            CheckContours(contour, frame)
            if CheckContours(contour, frame) :
                rect = cv.minAreaRect(contour)
                box = cv.boxPoints(rect)
                box = np.int32(box)
                
                width, height = int(rect[1][0]), int(rect[1][1])
                src_pts = box.astype("float32")

                dst_pts = np.array([[0, height-1],
                                    [0, 0],
                                    [width-1, 0],
                                    [width-1, height-1]], dtype="float32")
            
                M = cv.getPerspectiveTransform(src_pts, dst_pts)
                cropped = cv.warpPerspective(invert, M, (width, height))

                cv.drawContours(frame, [box], 0, (255 , 255 , 0), 2)

                


    return frame, cropped, box

def VictimDetection(frame):
    if frame is None or frame.size == 0:
        return None, None

    matrix_size = 9
    height, width = frame.shape[:2]
    scale = matrix_size / max(height, width)
    
    if scale == 0:
        return None, None
        
    resized = cv.resize(frame, (matrix_size, matrix_size), fx=scale, fy=scale)
    
    _, binary_matrix = cv.threshold(resized, 121, 1, cv.THRESH_BINARY)
    binary_matrix = binary_matrix.astype(np.uint8)
    
    best_match = None
    highest_similarity = -1
    
    for letter, matrices in patterns.items():
        for matrix in matrices:
            similarity = cosine_matrix_similarity(binary_matrix, np.array(matrix))
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = letter

    if highest_similarity >= 0.87:
        print(f"Best match: {best_match} (Similarity: {highest_similarity:.2f})")
    else:
        print("No strong match found", highest_similarity)
        print(np.array2string(binary_matrix, separator=', '))

    return best_match, binary_matrix, highest_similarity
    


while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break

    processed_frame = frame.copy()
    
    # color_detection ,cropped , color_found = ColorDetection(frame)
    cropped_vic = CropVictims(frame)[1]
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv.putText(processed_frame, f"FPS: {int(fps)}", (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # print(fps)

    # print(color_found)
    cv.imshow('color ',ColorDetection(frame)[0])
    # cv.imshow('frame', filters(frame))
    # cv.imshow('frame2 ', CropVictims(frame)[0])

    crop_frame, cropped_vic, box = CropVictims(frame.copy())
    if cropped_vic is not None:
        best_match, _ , similarity= VictimDetection(cropped_vic)

        if similarity > 0.87:
            if best_match and box is not None:
                cv.drawContours(processed_frame, [box], 0, (255, 0, 0), 2)

                cv.putText(processed_frame, best_match, (box[0][0], box[0][1] - 10), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                cv.putText(processed_frame, str(round(similarity, 2)), (box[0][0], box[0][1] - 35), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv.imshow('Processed Frame', processed_frame)
    
    if cropped_vic is not None:
        cv.imshow('color', cropped_vic)
    else:
        if cv.getWindowProperty('color', cv.WND_PROP_VISIBLE) >= 1:
            cv.destroyWindow('color')

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
