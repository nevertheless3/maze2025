import cv2 as cv
import numpy as np
import threading
import queue
import time
import json 
import os
from send_vicitm import SendVictim
import math
from sklearn.metrics.pairwise import cosine_similarity


victim_sender = SendVictim() 

COLOR_THRESHOLDS_FILE = r'C:\Users\Win11\Desktop\maze\maze-2026\maze2025\thresholds.json'

DEFAULT_COLOR_THRESHOLDS = {
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

def load_color_thresholds():
    if os.path.exists(COLOR_THRESHOLDS_FILE):
        with open(COLOR_THRESHOLDS_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return DEFAULT_COLOR_THRESHOLDS
    return DEFAULT_COLOR_THRESHOLDS


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
        self.color_thresholds = load_color_thresholds()
        self.detected_colors = []
        self.color_ranges = [
            {
                "name": "G",
                "lower": np.array(self.color_thresholds['green']['lower']),
                "upper": np.array(self.color_thresholds['green']['upper']),
                "display_color": (0, 255, 0)  
            },
            {
                "name": "R",
                "lower": np.array(self.color_thresholds['red']['lower']),
                "upper": np.array(self.color_thresholds['red']['upper']),
                "display_color": (0, 0, 255) 
            },
            {
                "name": "Y",
                "lower": np.array(self.color_thresholds['yellow']['lower']),
                "upper": np.array(self.color_thresholds['yellow']['upper']),
                "display_color": (0, 255, 255)  
            }
        ]
    def CheckColors(self , frame , cnt):
        H , W = frame.shape[:2]
        AREA = W * H
        area = cv.contourArea(cnt)
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int32(box)
        peri = cv.arcLength(cnt,True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
        w, h = int(rect[1][0]), int(rect[1][1])
        aspect_ratio = w/h if h >0 else 0

        # print('hhhheeeey' , area, AREA / 66)
        return (
                0.3 < aspect_ratio < 3 and
                3 < len(approx) < 7 and 
                AREA / 66 <= area <= AREA / 6
                )

    
    def check_parts(self, blob_image, color):
        
        height, width = blob_image.shape[:2]
        blob_image_LAB = cv.cvtColor(blob_image, cv.COLOR_BGR2LAB)

        block_width = width // 4
        block_height = height // 4

        thresh = next((c for c in self.color_ranges if c["name"] == color), None)
        
        blocks = [
            blob_image_LAB[i * block_height:(i + 1) * block_height, 
                           j * block_width:(j + 1) * block_width]
            for i in range(4) for j in range(4)
        ]
        means = np.array([np.mean(block, axis=(0, 1)) for block in blocks])
        
        lower = thresh['lower']
        upper = thresh['upper']
        matches = np.all((lower <= means) & (means <= upper), axis=1)
        
        return np.sum(matches) >= 14
    
    def detect(self, frame, wall_mask= None):
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
            if len(contours) == 0:
                continue
            largest_contour = max(contours, key=cv.contourArea)
            if self.CheckColors(frame , largest_contour):
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
                    victim_sender.FoundVictim(color["name"])

                    cv.drawContours(frame, [box], 0, color["display_color"], 2)
                    center = (int(rect[0][0]), int(rect[0][1]))
                    cv.circle(frame, center, 5, color["display_color"], -1)
        
        # print(self.color_thresholds)

        return frame, cropped
    

class VictimDetector:
    def __init__(self):

        N = np.nan
        self.patterns = {
            'H': [
                [[1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,1.,1.,1.,1.,1.,1.,1.,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.]],

                [[N , N , N , N , N , N , N , N , 2],
                [1., 1., 1., 1., 1., 1., 1., 1., 2],
                [0., 0., 0., 0., N , 0., 0., 0., 0],
                [0., 0., 0., 0., 1., 0., 0., 0., 0],
                [0., 0., 0., 0., 1., 0., 0., 0., 0],
                [0., 0., 0., 0., 1., 0., 0., 0., 0],
                [0., 0., 0., 0., 1., 0., 0., 0., 0],
                [N , 1., 1., 1., 1., 1., 1., 1., 1],
                [0., N , N , 1., 1., 1., N , N , 2]]
            ],
            'S': [
               [[N ,N ,N ,0.,0.,N ,N ,N ,0.],
                [N ,N ,N ,0.,N ,1.,N ,1.,N ],
                [N ,N ,0.,0.,N ,N ,0.,N ,N ],
                [1.,0.,0.,0.,N ,N ,0.,0.,1.],
                [1.,0.,0.,0.,1.,0.,0.,0.,1.],
                [1.,0.,0.,N ,1.,0.,0.,0.,1.],
                [N ,N ,0.,N ,N ,0.,0.,N ,1.],
                [N ,1.,N ,1.,N ,0.,N ,N ,N ],
                [0.,N ,N ,N ,0.,0.,N ,N ,N ]],

                [[0., 0., N , 0., 0., N , 1., N , 0.],
                [0., 1., N , 0., N , 1., 1., 1., 0.],
                [N , N , N , 0., N , N , 0., N , N ],
                [1., N , 0., 0., N , N , 0., N , 1.],
                [1., N , 0., N , N , N , 0., N , 1.],
                [N , N , 0., N , N , N , 0., N , 1.],
                [N , N , 0., N , N , 0., 0., N , 1.],
                [0., 1., 1., N , N , 0., N , 1., 0.],
                [0., N , 1., N , N , 0., N , N , 0.]]

            ],
            'U': [

                [[0.,N ,1.,1.,1.,1.,1.,1.,1.],
                [0.,N ,N ,N ,N ,N ,N ,N ,N ],
                [N ,N ,0.,0.,0.,0.,0.,0.,0.],
                [1.,0.,0.,0.,0.,0.,0.,0.,0.],
                [1.,0.,0.,0.,0.,0.,0.,0.,0.],
                [1.,0.,0.,0.,0.,0.,0.,0.,0.],
                [N ,N ,0.,0.,0.,0.,0.,0.,0.],
                [0.,N ,N ,N ,N ,N ,N ,N ,N ],
                [0.,N ,N ,1.,1.,1.,1.,1.,1.]],

                [[N ,1.,0.,0.,0.,0.,0.,1.,N],
                [N ,N ,0.,0.,0.,0.,0.,N ,N],
                [N ,N ,0.,0.,0.,0.,0.,N ,N ],
                [N ,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [N ,N ,0.,0.,0.,0.,0.,N ,N ],
                [N ,1.,N ,N ,N ,N ,N ,1.,0.],
                [0.,N ,N ,1.,1.,1.,N ,0.,0.]]

            ]
        }
    

    # @staticmethod
    # def cosine_similarity(vec1, vec2):
    #                               |>broadcasting
    #     return np.dot(vec1, vec2.T) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    
    def cosine_matrix_similarity(self , template, input_matrix):
        template = np.array(template, np.float32)

        valid_mask = ~np.isnan(template)
        t_vals = template[valid_mask]
        i_vals = input_matrix[valid_mask]
        
        t_vals = np.int8(t_vals)
        i_vals = np.int8(i_vals)
        
        flat1 = t_vals.flatten().reshape(1, -1)
        flat2 = i_vals.flatten().reshape(1, -1)
        
        return cosine_similarity(flat1,flat2)[0][0]

    
    def detect(self, frame):
        if frame is None or frame.size == 0:
            return None, None, None

        matrix_size = 9
        height, width = frame.shape[:2]
        scale = matrix_size / max(height, width)
        
        if scale == 0:
            return None, None, None
            
        resized = cv.resize(frame, (matrix_size, matrix_size), fx=scale, fy=scale)
        
        _, binary_matrix = cv.threshold(resized, 121, 1, cv.THRESH_BINARY)
        binary_matrix = binary_matrix.astype(np.float32)
        
        best_match = None
        highest_similarity = -1
        

        for letter, matrices in self.patterns.items():
            for matrix in matrices:
                num_rotations = 2 if letter == 'H' else 4
                
                for angle in range(num_rotations):
                    rotated_matrix = np.rot90(matrix, k=angle)
                    similarity = self.cosine_matrix_similarity(rotated_matrix, binary_matrix)
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = letter

        if 0.7 <highest_similarity < 0.98:
            print(np.array2string(binary_matrix, separator=', '))
            
        return best_match, binary_matrix, highest_similarity
 
class VictimCropper:
    def __init__(self):
        self.rng = np.random.RandomState(112244)
        self.SZ = 90

    def filters_crop(self , frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # gray = cv.GaussianBlur(gray , (31 ,31), 1)
        kernel = np.ones((3, 3), np.uint8)
        gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
        gray = cv.erode(gray, np.ones((3, 3), np.uint8))

        dst = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,53, 25)

        cv.imshow('filter', dst)
        return dst,gray


    def filters_detect(self , frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray , (31 ,31), 1)

        dst = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,53, 20)

        cv.imshow('jhjhjhjh', dst)
        return dst,gray

    def moments_check(self , cnt_moments):  

        hu_moments = np.zeros(7)
        Hu = [0,0,0,0,0,0,0]
        hu_moments = cv.HuMoments(cnt_moments).flatten()  # Assuming cnt_moments is the moments dict

        for m in range(7):
            Hu[m] = -1 * math.copysign(1.0, hu_moments[m]) * math.log10(abs(hu_moments[m])) * 100

        # S_check = (Hu[1] > 100 and Hu[1] < 225 and
        #         Hu[0] > 20 and Hu[0] < 70 and
        #         Hu[1] < Hu[3] and
        #         Hu[2] > 275 and Hu[2] < 700 and
        #         Hu[3] > 320 and Hu[3] < 600 and
        #         abs(Hu[5]) > 400 and
        #         abs(Hu[6]) > 600 and abs(Hu[6]) < 1200 and
        #         abs(Hu[6] / Hu[4]) > 0.75 and abs(Hu[6] / Hu[4]) < 1.3 and
        #         abs(Hu[5] / Hu[3]) >= 1.1 and abs(Hu[5] / Hu[3]) < 2.5 and
        #         abs(Hu[1] / Hu[0]) > 3.5 and abs(Hu[1] / Hu[0]) < 6)

        # hu_check = Hu[1] > 100

        nu_check = (cnt_moments['nu02'] > 0.11 and cnt_moments['nu02'] < 0.32 and
                    cnt_moments['nu20'] > 0.065 and cnt_moments['nu20'] < 0.32 and
                    abs(cnt_moments['nu11']) <= 0.09)

        # H_check = (Hu[1] > 100 and Hu[1] < 350 and
        #         Hu[0] < 60 and
        #         Hu[1] < Hu[3] and
        #         abs(Hu[5]) > 300 and
        #         abs(Hu[6]) > 480 and abs(Hu[6]) < 1200 and
        #         abs(Hu[6] / Hu[4]) > 0.75 and abs(Hu[6] / Hu[4]) < 1.3 and
        #         abs(Hu[1] / Hu[0]) > 3.6 and abs(Hu[1] / Hu[0]) < 8)

        # U_check = (Hu[1] > 100 and
        #         Hu[0] < 50 and
        #         abs(Hu[5]) > 300 and
        #         abs(Hu[6]) > 300 and abs(Hu[6]) < 1200 and
        #         abs(Hu[6] / Hu[4]) > 0.75 and abs(Hu[6] / Hu[4]) < 1.5 and
        #         abs(Hu[1] / Hu[0]) > 7 and abs(Hu[1] / Hu[0]) < 15)

        return nu_check # and (S_check or H_check or U_check)
    
    # def deskew(self , img):
    #     # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     img = 255 - img
    #     m = cv.moments(img)
    #     if abs(m['mu02']) < 1e-2:
    #         return img.copy()
    #     skew = m['mu11'] / m['mu02']
    #     M = np.float32([[1, skew, -0.7 * self.SZ * skew], [0, 1, 0]])
    #     out = cv.warpAffine(img, M, (self.SZ, self.SZ), flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
    #     mask = cv.inRange(out, 0, 0)
    #     out[mask == 255] = 255
    #     return out

    

    def crop(self , img, wall_mask = None):
        warped_fungus = None
        box = None

        dst, gray = self.filters_crop(img)
        contours, _ = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            cnt = contours[i]
            box = cv.boundingRect(cnt)
                            
            rect = cv.minAreaRect(cnt)
            area = cv.contourArea(cnt)

            hull = cv.convexHull(cnt)
            hull_area = cv.contourArea(hull)
            solidity = area / hull_area if hull_area != 0 else 0

            p = cv.boxPoints(rect)
            p = np.int32(p)

            rwidth = np.sqrt((p[1][0] - p[0][0])**2 + (p[1][1] - p[0][1])**2)
            rheight = np.sqrt((p[2][0] - p[1][0])**2 + (p[2][1] - p[1][1])**2)
            ar = rwidth / (rheight + 1e-9)
            rect_area = rwidth * rheight
            r = hull_area / (rect_area + 1e-9)

            if (box[0] > 1 and solidity <= 0.95 and solidity >= 0.32 and ar >= 0.3 and ar <= 3 and r >= 0.80 and area >= 200):
                color = (self.rng.randint(0, 256), self.rng.randint(0, 256), self.rng.randint(0, 256))
                colormat = np.full((150, 150, 3), color, dtype=np.uint8)

                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv.drawContours(mask, [cnt], -1, 255, -1)
                img_masked = np.full_like(img,(255,255,255))
                img_masked[mask == 255] = img[mask == 255]

                pts1 = p.astype(np.float32)
                pts2 = np.array([[0, 0], 
                                 [self.SZ- 1, 0], 
                                 [self.SZ- 1, self.SZ- 1], 
                                 [0, self.SZ- 1]], dtype=np.float32)
                
                prs = cv.getPerspectiveTransform(pts1, pts2)
                detect_img, _ = self.filters_detect(img)
                warped_fungus = cv.warpPerspective(detect_img, prs, (self.SZ, self.SZ))

                cv.imshow("jgfjgfjgjgj", warped_fungus  )


                cnt_Moments = cv.moments(cnt)

                cv.putText(img, str(cnt_Moments['nu02']), (p[0][0], p[0][1]), 1, 1, (255,0,0), 1, 1) 
                cv.putText(img, str(cnt_Moments['nu20']), (p[0][0], p[0][1]+ 20),1, 1, (0,255,0), 1, 1) 
                cv.putText(img, str(cnt_Moments['nu11']), (p[0][0], p[0][1]+ 30), 1, 1, (0,0,255), 1, 1) 

                # warped_fungus = self.deskew(warped_fungus)
                # filtered_warped_fungus = self.filters(warped_fungus)[0]
                # filtered_warped_fungus =cv.erode(filtered_warped_fungus, np.ones((5, 5), np.uint8))

                is_vic = self.moments_check(cnt_Moments)

                # warped_fungus = cv.resize(warped_fungus, (12, 12), interpolation=cv.INTER_NEAREST)

                cool = np.zeros_like(img)

                if is_vic:
                    tl = (box[0], box[1])
                    br = (box[0] + box[2], box[1] + box[3])
                    cv.rectangle(img, tl, br, color, 2)
                    cv.rectangle(cool, tl, br, color, -1)
                    cv.addWeighted(img, 1.0, cool, 0.3, 0.9, img)

                    return img , warped_fungus , p

                else:
                    colorr = (255, 255, 255)
                    tl = (box[0], box[1])
                    br = (box[0] + box[2], box[1] + box[3])
                    cv.rectangle(img, tl, br, colorr, 2)
                    cv.rectangle(cool, tl, br, colorr, -1)
                    cv.addWeighted(img, 1.0, cool, 0.3, -0.9, img)

        return img , None , None



    def Warp(self , cropped):
        if cropped is None:
            return None
        
        contours, _ = cv.findContours(cropped, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) < 1:
            return None
        
        contour = max(contours, key=cv.contourArea)

        if not contours:
            return cropped
        
        pts = contour.squeeze()
        x, y, w, h = cv.boundingRect(contour)

        if len(pts) < 4:
            return cropped

        sum_coords = pts.sum(axis=1)
        diff_coords = np.diff(pts, axis=1).reshape(-1)

        top_left = pts[np.argmin(sum_coords)]
        bottom_right = pts[np.argmax(sum_coords)]
        top_right = pts[np.argmin(diff_coords)]

        bottom_left = pts[np.argmax(diff_coords)]

        src_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

        width = max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
        height = max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))

        x_top_left , y_top_left = top_left
        x_top_right , y_top_right = top_right
        x_bottom_left , y_bottom_left = bottom_left
        x_bottom_right , y_bottom_right = bottom_right

        # print(x_top_left - x_bottom_left, x_top_right - x_bottom_right, y_bottom_left - y_bottom_right, y_top_left - y_top_right)
        if abs(x_top_left - x_bottom_left) < 5 and abs(x_top_right - x_bottom_right) < 5 and abs(y_bottom_left - y_bottom_right) < 5 and abs(y_top_left - y_top_right) < 5:
            final_warped = cropped

        else:
            scale = min(self.SZ / width, self.SZ / height) * 0.5

            dst_width, dst_height = width * scale, height * scale

            offset_x = (self.SZ - dst_width) // 2 
            offset_y = (self.SZ - dst_height) // 2 
            # print('offest' , offset_x , offset_y)
            dst_pts = np.array([
                [offset_x, offset_y],                        # Top-left
                [offset_x + dst_width, offset_y],           # Top-right
                [offset_x + dst_width, offset_y + dst_height],  # Bottom-right
                [offset_x, offset_y + dst_height]           # Bottom-left
            ], dtype="float32")

            M = cv.getPerspectiveTransform(src_corners, dst_pts)
            warped = cv.warpPerspective(cropped, M, (self.SZ, self.SZ))

            warped_contours , _ = cv.findContours(warped, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            contour = max(warped_contours, key=cv.contourArea, default = 0)
            
            if not warped_contours:
                return cropped
                
            
            x, y, w, h = cv.boundingRect(contour)

            final_warped = warped[y:y+h, x:x+w]
            final_warped = cv.resize(final_warped, (90, 90))

            moments = cv.moments(final_warped)

            is_vic = self.moments_check(moments)
        
            # if is_vic:
            #     return final_warped
            # final_warped = self.deskew(final_warped)

        return final_warped
            
class VideoCaptureThread(threading.Thread):
    def __init__(self, src=0, width=640, height=480, fps=120):
        threading.Thread.__init__(self)
        self.cap = cv.VideoCapture(src)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv.CAP_PROP_FPS, fps)
        self.frame_queue = queue.Queue(maxsize=2)  
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                time.sleep(0.001)  

    def stop(self):
        self.running = False
        self.cap.release()

class ProcessingThread(threading.Thread):
    def __init__(self, capture_thread):
        threading.Thread.__init__(self)
        self.capture_thread = capture_thread
        self.running = False
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.last_frame_time = time.time()
        
        self.wall_detector = WhiteWallDetector()
        self.color_detector = ColorDetector()
        self.victim_cropper = VictimCropper()
        self.victim_detector = VictimDetector()

    def run(self):
        self.running = True
        
        while self.running:
            if not self.capture_thread.frame_queue.empty():
                frame = self.capture_thread.frame_queue.get()
                current_time = time.time()

                frame_time = current_time - self.last_frame_time
                self.last_frame_time = current_time

                wall_mask = self.wall_detector.detect(frame)
                color_frame, _ = self.color_detector.detect(frame.copy(), wall_mask)
                victim_frame, cropped_vic, box = self.victim_cropper.crop(color_frame, wall_mask)
                warped_vic = self.victim_cropper.Warp(cropped_vic)
                # filters = self.victim_cropper.filters(frame)

                if warped_vic is not None:
                    best_match, _, similarity = self.victim_detector.detect(warped_vic)
                    if best_match and box is not None:
                        # cv.drawContours(victim_frame, [box], 0, (255,0,0), 1)
                        cv.putText(victim_frame, f"{similarity:.2f}", 
                                   (box[0][0], box[0][1]-35), 
                                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                        
                        if similarity >= 0.8:
                            victim_sender.FoundVictim(best_match)
                            cv.putText(victim_frame, best_match, 
                                       (box[0][0], box[0][1]-10), 
                                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,0), 2)
                
                self.frame_count += 1
                if current_time - self.last_fps_time >= 1:
                    self.fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time

                cv.putText(victim_frame, f"FPS: {int(self.fps)}", (10, 30),
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # cv.imshow('filter', filters)
                cv.imshow('Detection', victim_frame)
                if cropped_vic is not None:
                    cv.imshow('cropped', cropped_vic)
                else:
                    if cv.getWindowProperty('cropped', cv.WND_PROP_VISIBLE) >= 1:
                        cv.destroyWindow('cropped')

                if warped_vic is not None:
                    cv.imshow('warped', warped_vic)
                else:
                    if cv.getWindowProperty('warped', cv.WND_PROP_VISIBLE) >= 1:
                        cv.destroyWindow('warped')


                if cv.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

    def stop(self):
        self.running = False

if __name__ == "__main__":
    CAMERA_WIDTH = 160
    CAMERA_HEIGHT = 120
    TARGET_FPS = 30

    cap_thread = VideoCaptureThread(
        src=0, 
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=TARGET_FPS
    )
    
    proc_thread = ProcessingThread(cap_thread)

    try:
        cap_thread.start()
        proc_thread.start()
        
        while proc_thread.running:
            time.sleep(0.1)  
            
    except KeyboardInterrupt:
        pass
    finally:
        proc_thread.stop()
        cap_thread.stop()
        proc_thread.join()
        cap_thread.join()
        cv.destroyAllWindows()