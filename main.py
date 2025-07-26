import cv2 as cv
import numpy as np
import threading
import queue
import time
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Core Detection Classes (Unchanged from your original)
# ----------------------------
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
    
    def CheckColors(self , cnt):
        area = cv.contourArea(cnt)
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int32(box)
        peri = cv.arcLength(cnt,True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
        w, h = int(rect[1][0]), int(rect[1][1])
        aspect_ratio = w/h if h >0 else 0

        return ( 1000< area < 9000 and
                0.3 < aspect_ratio < 3 and
                 3 < len(approx) < 7 )
    
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
            if self.CheckColors(largest_contour):
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

                [[N,1.,1.,1.,1.,1.,1.,1.,1.],
                [N ,N ,N ,N ,1.,N ,N ,N ,N ],
                [0.,0.,0.,0.,1.,0.,0.,0.,0.],
                [0.,0.,0.,0.,1.,0.,0.,0.,0.],
                [0.,0.,0.,0.,1.,0.,0.,0.,0.],
                [0.,0.,0.,0.,1.,0.,0.,0.,0.],
                [0.,0.,0.,0.,1.,0.,0.,0.,0.],
                [N ,N ,N ,N ,1.,N ,N ,N ,N ],
                [N ,1.,1.,1.,1.,1.,1.,1.,1.]]

            ],
            'S': [
                [[0.,N ,1.,1.,1.,1.,1.,N ,N ],
                [N ,1.,N ,0.,0.,0.,N ,N ,N ],
                [N ,N ,0.,0.,0.,0.,0.,N ,N ],
                [N ,1.,N ,N ,0.,0.,0.,0.,0.],
                [0.,N ,N ,1.,1.,1.,N ,N ,0.],
                [0.,0.,0.,0.,0.,N ,N ,1.,N ],
                [N ,N ,0.,0.,0.,0.,0.,N ,N ],
                [N ,N ,N ,0.,0.,0.,N ,1.,N ],
                [N ,N ,1.,1.,1.,1.,1.,N ,0.]],

               [[N ,N ,N ,0.,0.,N ,N ,N ,0.],
                [N ,N ,N ,0.,N ,1.,N ,1.,N ],
                [N ,N ,0.,0.,N ,N ,0.,N ,1.],
                [1.,0.,0.,0.,N ,N ,0.,0.,1.],
                [1.,0.,0.,0.,1.,0.,0.,0.,1.],
                [1.,0.,0.,N ,1.,0.,0.,0.,1.],
                [N ,N ,0.,N ,N ,0.,0.,N ,1.],
                [N ,1.,N ,1.,N ,0.,N ,N ,N ],
                [0.,N ,N ,N ,0.,0.,N ,N ,N ]]


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

                [[1.,1.,1.,1.,1.,1.,1.,N ,0.],
                [N ,N ,N ,N ,N ,N ,N ,N ,N ],
                [0.,0.,0.,0.,0.,0.,0.,N ,1.],
                [0.,0.,0.,0.,0.,0.,0.,0.,1.],
                [0.,0.,0.,0.,0.,0.,0.,0.,1.],
                [0.,0.,0.,0.,0.,0.,0.,0.,1.],
                [0.,0.,0.,0.,0.,0.,0.,0.,1.],
                [N ,N ,N ,N ,N ,N ,N ,N ,N ],
                [1.,1.,1.,1.,1.,1.,1.,N ,0.]],

                [[0.,0.,1.,1.,1.,1.,1.,N ,0.],
                [N ,1.,N ,0.,0.,0.,N ,1.,N ],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [N ,N ,0.,0.,0.,0.,0.,N ,1.]],


                [[1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [1.,N ,0.,0.,0.,0.,0.,N ,1.],
                [N ,N ,0.,0.,0.,0.,0.,N ,1.],
                [N ,N ,N ,0.,0.,0.,N ,N ,N ],
                [0.,N ,1.,1.,1.,1.,1.,N ,0.]]

            ]
        }
    
    @staticmethod
    def cosine_matrix_similarity(template, input_matrix):
        template = np.array(template, np.float16)

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
        binary_matrix = binary_matrix.astype(np.float16)
        
        best_match = None
        highest_similarity = -1
        
        for letter, matrices in self.patterns.items():
            for matrix in matrices:
                similarity = self.cosine_matrix_similarity(np.array(matrix), binary_matrix)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = letter
        
        if 0.94 <highest_similarity < 0.98:
            print(np.array2string(binary_matrix, separator=', '))
        # highest_similarity *= 10**4
        
        return best_match, binary_matrix, highest_similarity
 
class VictimCropper:
    def __init__(self):
        pass

    def filters(self , frame):
        gray = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray , (31 ,31), 1)
        adaptive_mean = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 53, 20)
        invert = (255 - adaptive_mean)

        return invert
    
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

        invert = self.filters(frame)

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
        self.last_time = 0
        
        self.wall_detector = WhiteWallDetector()
        self.color_detector = ColorDetector()
        self.victim_cropper = VictimCropper()
        self.victim_detector = VictimDetector()

    def run(self):
        self.running = True
        self.last_time = time.time()
        
        while self.running:
            if not self.capture_thread.frame_queue.empty():
                frame = self.capture_thread.frame_queue.get()

                wall_mask = self.wall_detector.detect(frame)
                color_frame, _ = self.color_detector.detect(frame.copy(), wall_mask)
                victim_frame, cropped_vic, box = self.victim_cropper.crop(color_frame, wall_mask)

                if cropped_vic is not None:
                    best_match, _, similarity = self.victim_detector.detect(cropped_vic)
                    if best_match and box is not None:
                        cv.drawContours(victim_frame, [box], 0, (255,0,0), 1)
                        cv.putText(victim_frame, f"{similarity:.2f}", 
                                   (box[0][0], box[0][1]-35), 
                                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                        
                        if similarity >= 0.95:
                            cv.putText(victim_frame, best_match, 
                                       (box[0][0], box[0][1]-10), 
                                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,0), 2)
                
                curr_time = time.time()
                self.fps = 1 / (curr_time - self.last_time)
                self.last_time = curr_time
                cv.putText(victim_frame, f"FPS: {int(self.fps)}", (10, 30),
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv.imshow('Detection', victim_frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

    def stop(self):
        self.running = False

if __name__ == "__main__":
    CAMERA_WIDTH = 640  
    CAMERA_HEIGHT = 480
    TARGET_FPS = 120    

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