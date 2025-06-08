import cv2 as cv
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import chi2_kernel

class WhiteWallDetector:
    def __init__(self):
        self.lower_white = np.array([0, 0, 43])  
        self.upper_white = np.array([180, 77, 255])
        
    def detect(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_white, self.upper_white)
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            if cv.contourArea(largest_contour) > 100:
                final_mask = np.zeros_like(mask)
                cv.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv.FILLED)
                return final_mask
        return None

class ColorDetector:
    def __init__(self):
        self.color_ranges = [
            {
                "name": "Green",
                "lower": np.array([0,49,87]),
                "upper": np.array([100, 125, 150]),
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
        self.detected_colors = []
    
    def check_parts(self, blob_image, color):
        height, width = blob_image.shape[:2]
        blob_image_LAB = cv.cvtColor(blob_image, cv.COLOR_BGR2LAB)

        block_width = width // 4
        block_height = height // 4

        thresh = None
        for c in self.color_ranges:
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
                
                block = blob_image_LAB[y1:y2, x1:x2]
                    
                l_mean = np.mean(block[:,:,0])
                a_mean = np.mean(block[:,:,1])
                b_mean = np.mean(block[:,:,2])

                if (thresh['lower'][0] <= l_mean <= thresh['upper'][0] and
                    thresh['lower'][1] <= a_mean <= thresh['upper'][1] and
                    thresh['lower'][2] <= b_mean <= thresh['upper'][2]):
                    count += 1
        
        return count >= 14
    
    def detect(self, frame, wall_mask=None):
        cropped = None 

        self.detected_colors = []
        lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
        
        if wall_mask is not None:
            lab = cv.bitwise_and(lab, lab, mask=wall_mask)
            
        for color in self.color_ranges:
            mask = cv.inRange(lab, color["lower"], color["upper"])
            

            kernel = np.ones((5,5), np.uint8)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
            
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest_contour = max(contours, key=cv.contourArea)
            if cv.contourArea(largest_contour) > 100:  

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
                
         
                if self.check_parts(cropped, color["name"]):
                    color_found = True
                    self.detected_colors.append(color["name"])

                    cv.drawContours(frame, [box], 0, color["display_color"], 2)
                    center = (int(rect[0][0]), int(rect[0][1]))
                    cv.circle(frame, center, 5, color["display_color"], -1)

        return frame, cropped
    

class VictimDetector:
    def __init__(self):
        self.patterns = {
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
            ],
            'S': [
                [[0, 0, 0, 0, 0, 0, 1, 1, 0],
                [1, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 0, 1, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 0, 0, 0, 0, 1],
                [0, 1, 1, 1, 0, 0, 0, 0, 1]],

                [[0, 1, 0, 0, 0, 0, 1, 1, 0],
                [1, 1, 0, 0, 0, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 0, 0, 0, 1],
                [0, 1, 1, 1, 0, 0, 0, 1, 1],
                [0, 0, 1, 0, 0, 0, 0, 1, 0]],

                [[0, 0, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 0]],

                [[0, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0]],

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
            'U': [
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
            ]
        }
    
    @staticmethod
    def cosine_matrix_similarity(mat1, mat2):
        flat1 = mat1.flatten().reshape(1, -1)
        flat2 = mat2.flatten().reshape(1, -1)
        return chi2_kernel(flat1, flat2, gamma=0.009)[0][0]

    def filters(self , frame):
        gray = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)
        # gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)
        blur = cv.GaussianBlur(gray , (15 ,15), 1)
        adaptive_mean = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 53, 20)
        invert = (255 - adaptive_mean)

        return gray
    
    def detect(self, frame):
        if frame is None or frame.size == 0:
            return None, None, None

        matrix_size = 9
        height, width = frame.shape[:2]
        scale = matrix_size / max(height, width) if max(height , width) > 0 else 0
        
        if scale == 0:
            return None, None, None
            
        resized = cv.resize(frame, (matrix_size, matrix_size), fx=scale, fy=scale)
        
        _, binary_matrix = cv.threshold(resized, 121, 1, cv.THRESH_BINARY)
        binary_matrix = binary_matrix.astype(np.uint8)
        
        best_match = None
        highest_similarity = -1
        
        for letter, matrices in self.patterns.items():
            for matrix in matrices:
                similarity = self.cosine_matrix_similarity(binary_matrix, np.array(matrix))
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = letter
        
        if highest_similarity <= 9:
            print(np.array2string(binary_matrix, separator=', '))
        # highest_similarity *= 10**4
        
        return best_match, binary_matrix, highest_similarity
 
class VictimCropper:
    def __init__(self):
        pass
    
    def check_contours(self, cnt, frame):
        max_area = 9000
        min_area = 100
        height, width = frame.shape[:2]
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int32(box)
        
        w, h = int(rect[1][0]), int(rect[1][1])
        area = cv.contourArea(cnt)
        x, y, _, _ = cv.boundingRect(cnt)
        aspect_ratio = w/h if h > 0 else 0

        return (
            min_area <= area <= max_area and
            width//16 <= w <= width//2.2 and
            0.55 <= aspect_ratio <= 1.8 and
            x >= 2 and
            x + w <= width - 2 and
            y >= 8 and
            h + y <= height 
        )
    
    def crop(self, frame, wall_mask = None):
        cropped = None
        box = None

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (15, 15), 1)
        adaptive_mean = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv.THRESH_BINARY, 201, 16)
        invert = (255 - adaptive_mean)

        if wall_mask is not None:
            invert = cv.bitwise_and(invert, invert, mask=wall_mask)

        contours, _ = cv.findContours(invert.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if contours:
            contours = sorted(contours , key= cv.contourArea , reverse= True)[:3]
            
            for contour in contours:
                if self.check_contours(contour, frame):
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

                    cv.drawContours(frame, [box], 0, (255, 255, 0), 2)

        return frame, cropped, box
    

class VideoProcessor:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video capture")
            
        self.wall_detector = WhiteWallDetector()
        self.color_detector = ColorDetector()
        self.victim_cropper = VictimCropper()
        self.victim_detector = VictimDetector()
        
        self.prev_time = 0
        self.curr_time = 0
    
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, False
                    

        wall_mask = self.wall_detector.detect(frame)
        
        color_frame = self.color_detector.detect(frame.copy(), wall_mask)
        
        victim_frame, cropped_vic, box = self.victim_cropper.crop(frame.copy(), wall_mask)
        
        self.curr_time = time.time()
        fps = 1 / (self.curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = self.curr_time
        cv.putText(victim_frame, f"FPS: {int(fps)}", (10,30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        

        if cropped_vic is not None:
            best_match, _, similarity = self.victim_detector.detect(cropped_vic)
            if best_match and box is not None:
                cv.drawContours(victim_frame, [box], 0, (255,0,0), 1)
                cv.putText(victim_frame, f"{similarity:.2f}", 
                          (box[0][0], box[0][1]-35), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                
                if similarity > 0.9:
                    cv.putText(victim_frame, best_match, 
                              (box[0][0], box[0][1]-10), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,0), 2)
                else:
                    cv.putText(victim_frame, 'rejected', 
                              (box[0][0], box[0][1]-10), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        
        return victim_frame, wall_mask , True
    
    def release(self):
        self.cap.release()
        cv.destroyAllWindows()

def main():
    try:
        processor = VideoProcessor()
        while True:
            frame, wall_mask, success = processor.process_frame()
            if not success:
                break
                
            cv.imshow('Detection', frame)
            cv.imshow('mask', wall_mask)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        processor.release()

if __name__ == "__main__":
    main()