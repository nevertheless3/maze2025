import cv2 as cv
import numpy as np 
import os
import json

CONFIG_FILE = r'C:\Users\Win11\Desktop\maze\maze-2026\maze2025\thresholds.json'

DEFAULT_VALS = {
    "green": {
        "lower": [0, 17, 128],
        "upper": [206, 118, 168]
    },
    "yellow": {
        "lower": [0, 118, 173],
        "upper": [255, 144, 255]
    },
    "red": {
        "lower": [0, 153, 139],
        "upper": [255, 255, 199]
    }
}


def LoadVals():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            try:
                saved = json.load(f)
                for color in DEFAULT_VALS:
                    if color not in saved:
                        saved[color] = DEFAULT_VALS[color]
                    else:
                        if 'lower' not in saved[color]:
                            saved[color]['lower'] = DEFAULT_VALS[color]['lower']
                        if 'upper' not in saved[color]:
                            saved[color]['upper'] = DEFAULT_VALS[color]['upper']
                return saved
            except json.JSONDecodeError:
                return DEFAULT_VALS
    return DEFAULT_VALS

def SaveVals(lower, upper, color):
    all_colors = LoadVals()
    all_colors[color] = {
        'lower': [int(x) for x in lower],
        'upper': [int(x) for x in upper]
    }
    
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(all_colors, f, indent=4)
    print(f"Saved {color} thresholds: lower={lower}, upper={upper}")

def ask_color():
    print("\nWhich color are you saving?")
    print("1. Green")
    print("2. Yellow")
    print("3. Red")
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            return {'1': 'green', '2': 'yellow', '3': 'red'}[choice]
        print("Invalid choice. Please enter 1, 2, or 3")

window_name = 'LAB Thresholding'
cv.namedWindow(window_name)
all_colors = LoadVals()
current_color = 'green'
current_lower = all_colors[current_color]['lower']
current_upper = all_colors[current_color]['upper']

cv.createTrackbar('L lower', window_name, current_lower[0], 255, lambda x: None)
cv.createTrackbar('A lower', window_name, current_lower[1], 255, lambda x: None)
cv.createTrackbar('B lower', window_name, current_lower[2], 255, lambda x: None)

cv.createTrackbar('L upper', window_name, current_upper[0], 255, lambda x: None)
cv.createTrackbar('A upper', window_name, current_upper[1], 255, lambda x: None)
cv.createTrackbar('B upper', window_name, current_upper[2], 255, lambda x: None)

cap = cv.VideoCapture(0)

try:
    while True:  
        ret, frame = cap.read()  
        if not ret:  
            break 

        lower = [
            cv.getTrackbarPos('L lower', window_name),
            cv.getTrackbarPos('A lower', window_name),
            cv.getTrackbarPos('B lower', window_name)
        ]
        upper = [
            cv.getTrackbarPos('L upper', window_name),
            cv.getTrackbarPos('A upper', window_name),
            cv.getTrackbarPos('B upper', window_name)
        ]

        lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
        lower_array = np.array(lower)
        upper_array = np.array(upper)
        mask = cv.inRange(lab, lower_array, upper_array)
        result = cv.bitwise_and(frame, frame, mask=mask)

        cv.imshow('Original', frame)
        cv.imshow('LAB', lab)
        cv.imshow('Mask', mask)
        cv.imshow('Result', result)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            color = ask_color()
            SaveVals(lower, upper, color)
            break
        elif key == ord('s'):
            color = ask_color()
            SaveVals(lower, upper, color)
        elif key == ord('1'):
            current_color = 'green'
            current_lower = all_colors[current_color]['lower']
            current_upper = all_colors[current_color]['upper']
            cv.setTrackbarPos('L lower', window_name, current_lower[0])
            cv.setTrackbarPos('A lower', window_name, current_lower[1])
            cv.setTrackbarPos('B lower', window_name, current_lower[2])
            cv.setTrackbarPos('L upper', window_name, current_upper[0])
            cv.setTrackbarPos('A upper', window_name, current_upper[1])
            cv.setTrackbarPos('B upper', window_name, current_upper[2])
        elif key == ord('2'):
            current_color = 'yellow'
            current_lower = all_colors[current_color]['lower']
            current_upper = all_colors[current_color]['upper']
            cv.setTrackbarPos('L lower', window_name, current_lower[0])
            cv.setTrackbarPos('A lower', window_name, current_lower[1])
            cv.setTrackbarPos('B lower', window_name, current_lower[2])
            cv.setTrackbarPos('L upper', window_name, current_upper[0])
            cv.setTrackbarPos('A upper', window_name, current_upper[1])
            cv.setTrackbarPos('B upper', window_name, current_upper[2])
        elif key == ord('3'):
            current_color = 'red'
            current_lower = all_colors[current_color]['lower']
            current_upper = all_colors[current_color]['upper']
            cv.setTrackbarPos('L lower', window_name, current_lower[0])
            cv.setTrackbarPos('A lower', window_name, current_lower[1])
            cv.setTrackbarPos('B lower', window_name, current_lower[2])
            cv.setTrackbarPos('L upper', window_name, current_upper[0])
            cv.setTrackbarPos('A upper', window_name, current_upper[1])
            cv.setTrackbarPos('B upper', window_name, current_upper[2])

finally:
    cap.release()
    cv.destroyAllWindows()