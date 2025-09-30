import cv2 as cv
import numpy as np
import threading
import queue
import sys
import time
import json
import os
import subprocess
import re
# noinspection PyUnresolvedReferences
from picamera2 import Picamera2
from send_victim import SendVictim
import psutil
import multiprocessing as mp

use_imshow = True
if len(sys.argv) > 1:
    use_imshow = sys.argv[1] == 'enable'
    print("Imshow enable" if use_imshow else "Imshow disabled")

victim_sender = SendVictim()

COLOR_THRESHOLDS_FILE = 'thresholds.json'

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


def get_camera_map():
    result = subprocess.run(
        ["rpicam-hello", "--list-cameras"],
        capture_output=True, text=True
    )

    cam_map = {}
    for line in result.stdout.splitlines():
        match = re.match(r"(\d+)\s*:\s*(\S+).*(i2c@\d+)", line)
        if match:
            index, sensor, bus = match.groups()

            if bus == "i2c@88000":
                cam_map["R"] = int(index)
            elif bus == "i2c@80000":
                cam_map["L"] = int(index)
    return cam_map


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

        kernel = np.ones((5, 5), np.uint8)
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

    def CheckColors(self, frame, cnt):
        H, W = frame.shape[:2]
        AREA = W * H
        area = cv.contourArea(cnt)
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int32(box)
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
        w, h = int(rect[1][0]), int(rect[1][1])
        aspect_ratio = w / h if h > 0 else 0

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

    def detect(self, frame, wall_mask=None):
        cropped = None

        self.detected_colors = []
        lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)

        if wall_mask is not None:
            lab = cv.bitwise_and(lab, lab, mask=wall_mask)

        for color in self.color_ranges:
            mask = cv.inRange(lab, color["lower"], color["upper"])

            kernel = np.ones((5, 5), np.uint8)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            largest_contour = max(contours, key=cv.contourArea)
            if self.CheckColors(frame, largest_contour):
                rect = cv.minAreaRect(largest_contour)
                box = cv.boxPoints(rect)
                box = np.int32(box)

                width, height = int(rect[1][0]), int(rect[1][1])
                src_pts = box.astype("float32")

                dst_pts = np.array([[0, height - 1],
                                    [0, 0],
                                    [width - 1, 0],
                                    [width - 1, height - 1]], dtype="float32")

                M = cv.getPerspectiveTransform(src_pts, dst_pts)
                cropped = cv.warpPerspective(frame, M, (width, height))

                if self.check_parts(cropped, color["name"]):
                    color_found = True
                    self.detected_colors.append(color["name"])
                    victim_sender.FoundVictim('right', color["name"])

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
                [[1., N, 0., 0., 0., 0., 0., N, 1.],
                 [1., N, 0., 0., 0., 0., 0., N, 1.],
                 [1., N, 0., 0., 0., 0., 0., N, 1.],
                 [1., N, 0., 0., 0., 0., 0., N, 1.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [1., N, 0., 0., 0., 0., 0., N, 1.],
                 [1., N, 0., 0., 0., 0., 0., N, 1.],
                 [1., N, 0., 0., 0., 0., 0., N, 1.],
                 [1., N, 0., 0., 0., 0., 0., N, 1.]],

                [[N, N, N, N, N, N, N, N, 2],
                 [1., 1., 1., 1., 1., 1., 1., 1., 2],
                 [0., 0., 0., 0., N, 0., 0., 0., 0],
                 [0., 0., 0., 0., 1., 0., 0., 0., 0],
                 [0., 0., 0., 0., 1., 0., 0., 0., 0],
                 [0., 0., 0., 0., 1., 0., 0., 0., 0],
                 [0., 0., 0., 0., 1., 0., 0., 0., 0],
                 [N, 1., 1., 1., 1., 1., 1., 1., 1],
                 [0., N, N, 1., 1., 1., N, N, 2]]
            ],
            'S': [
                [[N, N, N, 0., 0., N, N, N, 0.],
                 [N, N, N, 0., N, 1., N, 1., N],
                 [N, N, 0., 0., N, N, 0., N, N],
                 [1., 0., 0., 0., N, N, 0., 0., 1.],
                 [1., 0., 0., 0., 1., 0., 0., 0., 1.],
                 [1., 0., 0., N, 1., 0., 0., 0., 1.],
                 [N, N, 0., N, N, 0., 0., N, 1.],
                 [N, 1., N, 1., N, 0., N, N, N],
                 [0., N, N, N, 0., 0., N, N, N]],

                [[0., 0., N, 0., 0., N, 1., N, 0.],
                 [0., 1., N, 0., N, 1., 1., 1., 0.],
                 [N, N, N, 0., N, N, 0., N, N],
                 [1., N, 0., 0., N, N, 0., N, 1.],
                 [1., N, 0., N, N, N, 0., N, 1.],
                 [N, N, 0., N, N, N, 0., N, 1.],
                 [N, N, 0., N, N, 0., 0., N, 1.],
                 [0., 1., 1., N, N, 0., N, 1., 0.],
                 [0., N, 1., N, N, 0., N, N, 0.]]

            ],
            'U': [

                [[0., N, 1., 1., 1., 1., 1., 1., 1.],
                 [0., N, N, N, N, N, N, N, N],
                 [N, N, 0., 0., 0., 0., 0., 0., 0.],
                 [1., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [1., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [1., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [N, N, 0., 0., 0., 0., 0., 0., 0.],
                 [0., N, N, N, N, N, N, N, N],
                 [0., N, N, 1., 1., 1., 1., 1., 1.]],

                [[N, 1., 0., 0., 0., 0., 0., 1., N],
                 [N, N, 0., 0., 0., 0., 0., N, N],
                 [N, N, 0., 0., 0., 0., 0., N, N],
                 [N, N, 0., 0., 0., 0., 0., N, 1.],
                 [1., N, 0., 0., 0., 0., 0., N, 1.],
                 [1., N, 0., 0., 0., 0., 0., N, 1.],
                 [N, N, 0., 0., 0., 0., 0., N, N],
                 [N, 1., N, N, N, N, N, 1., 0.],
                 [0., N, N, 1., 1., 1., N, 0., 0.]]

            ]
        }

    @staticmethod
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2.T) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def cosine_matrix_similarity(self, template, input_matrix):
        template = np.array(template, np.float32)

        valid_mask = ~np.isnan(template)
        t_vals = template[valid_mask]
        i_vals = input_matrix[valid_mask]

        t_vals = np.int8(t_vals)
        i_vals = np.int8(i_vals)

        flat1 = t_vals.flatten().reshape(1, -1)
        flat2 = i_vals.flatten().reshape(1, -1)

        return self.cosine_similarity(flat1, flat2)[0][0]

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

        if 0.7 < highest_similarity < 0.98:
            print(np.array2string(binary_matrix, separator=', '))

        return best_match, binary_matrix, highest_similarity


class VictimCropper:
    def __init__(self):
        self.SIZE = 90

    def filters(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (31, 31), 1)
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
        aspect_ratio = w / h if h > 0 else 0

        return (
                min_area <= area <= max_area and
                width // 16 <= w <= width // 2.2 and
                0.55 <= aspect_ratio <= 1.8 and
                x >= 2 and
                x + w <= width - 2 and
                y >= 8 and
                h + y <= height
        )

    def crop(self, frame, wall_mask=None):
        cropped = None
        box = None

        invert = self.filters(frame)

        if wall_mask is not None:
            invert = cv.bitwise_and(invert, invert, mask=wall_mask)

        contours, _ = cv.findContours(invert.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if contours:
            contours = sorted(contours, key=cv.contourArea, reverse=True)[:3]

            for contour in contours:
                if self.check_contours(contour, frame):
                    rect = cv.minAreaRect(contour)
                    box = cv.boxPoints(rect)
                    box = np.int32(box)

                    width, height = int(rect[1][0]), int(rect[1][1])
                    src_pts = box.astype("float32")

                    dst_pts = np.array([
                        [0, 0],
                        [self.SIZE - 1, 0],
                        [self.SIZE - 1, self.SIZE - 1],
                        [0, self.SIZE - 1]
                    ], dtype="float32")

                    M = cv.getPerspectiveTransform(src_pts, dst_pts)
                    cropped = cv.warpPerspective(invert, M, (self.SIZE, self.SIZE))

                    cv.drawContours(frame, [box], 0, (255, 255, 0), 2)

        return frame, cropped, box

    def Warp(self, cropped):
        if cropped is None:
            return None

        contours, _ = cv.findContours(cropped, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

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

        x_top_left, y_top_left = top_left
        x_top_right, y_top_right = top_right
        x_bottom_left, y_bottom_left = bottom_left
        x_bottom_right, y_bottom_right = bottom_right

        # print(x_top_left - x_bottom_left, x_top_right - x_bottom_right, y_bottom_left - y_bottom_right, y_top_left - y_top_right)
        if abs(x_top_left - x_bottom_left) < 5 and abs(x_top_right - x_bottom_right) < 5 and abs(y_bottom_left - y_bottom_right) < 5 and abs(
                y_top_left - y_top_right) < 5:
            final_warped = cropped

        else:
            scale = min(self.SIZE / width, self.SIZE / height) * 0.5

            dst_width, dst_height = width * scale, height * scale

            offset_x = (self.SIZE - dst_width) // 2
            offset_y = (self.SIZE - dst_height) // 2
            # print('offest' , offset_x , offset_y)
            dst_pts = np.array([
                [offset_x, offset_y],  # Top-left
                [offset_x + dst_width, offset_y],  # Top-right
                [offset_x + dst_width, offset_y + dst_height],  # Bottom-right
                [offset_x, offset_y + dst_height]  # Bottom-left
            ], dtype="float32")

            M = cv.getPerspectiveTransform(src_corners, dst_pts)
            warped = cv.warpPerspective(cropped, M, (self.SIZE, self.SIZE))

            warped_contours, _ = cv.findContours(warped, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            contour = max(warped_contours, key=cv.contourArea, default=0)

            if not warped_contours:
                return cropped

            x, y, w, h = cv.boundingRect(contour)

            final_warped = warped[y:y + h, x:x + w]
            final_warped = cv.resize(final_warped, (28, 28))

        return final_warped


#
# class VideoCaptureThread(threading.Thread):
#     def __init__(self, src=0, width=160, height=120, fps=120):
#         threading.Thread.__init__(self)
#         cam_id = get_camera_map()["L"]
#         self.picam = Picamera2(cam_id)
#         self.picam_config = self.picam.create_preview_configuration(
#             main={"size": (width, height), "format": "RGB888"},
#             buffer_count=8,
#         )
#         self.picam_config["controls"]["FrameRate"] = fps  # Set FPS
#         self.picam.configure(self.picam_config)
#         self.picam.set_controls(
#             {
#                 # "AeEnable": False,
#                 "ExposureTime": 20000,  # 10 ms
#                 # "AnalogueGain": 1.0
#             }
#         )
#         # self.picam.set_controls({"ExposureTime": 20000})
#
#         self.picam.start()
#         self.frame_queue = queue.Queue(maxsize=2)
#         self.running = False
#
#     def run(self):
#         self.running = True
#         loop_time = time.time()
#         while self.running:
#             if not self.frame_queue.full():
#                 # t = time.time()
#                 # print("Loop Time:", (t - loop_time) * 1000, end=", ")
#                 # loop_time = t
#                 frame = self.picam.capture_array()
#                 # print("Frame time:", (time.time() - t) * 1000, end=", ")
#                 # t1 = time.time()
#                 self.frame_queue.put(frame)
#                 # print("Queue time:", (time.time() - t1)*1000)
#             else:
#                 # print("skip")
#                 time.sleep(0.001)
#
#     def stop(self):
#         self.running = False
#         self.picam.stop()
#
#
# class ProcessingThread(threading.Thread):
#     def __init__(self, capture_thread):
#         threading.Thread.__init__(self)
#         self.capture_thread = capture_thread
#         self.running = False
#         self.fps = 0
#         self.frame_count = 0
#
#         self.wall_detector = WhiteWallDetector()
#         self.color_detector = ColorDetector()
#         self.victim_cropper = VictimCropper()
#         self.victim_detector = VictimDetector()
#
#         cam_id = get_camera_map()["L"]
#         self.picam = Picamera2(cam_id)
#         self.picam_config = self.picam.create_preview_configuration(
#             main={"size": (160, 120), "format": "RGB888"},
#             buffer_count=8,
#         )
#         self.picam_config["controls"]["FrameRate"] = 200  # Set FPS
#         self.picam.configure(self.picam_config)
#         self.picam.set_controls(
#             {
#                 # "AeEnable": False,
#                 "ExposureTime": 20000,  # 20 ms
#                 "NoiseReductionMode": 1, # Fast. 0: Off, 1: Fast, 2: HighQuality, 3: Minimal, 4. ZeroShutterLag
#                 # "AnalogueGain": 1.0}
#             }
#         )
#         # self.picam.set_controls({"ExposureTime": 20000})
#
#         self.picam.start()
#         self.frame_queue = queue.Queue(maxsize=2)
#
#     def run(self):
#         self.running = True
#         prev_time = time.time()
#         frame_count = 0
#         while self.running:
#             # if not self.capture_thread.frame_queue.empty():
#             t1 = time.time()
#
#             frame = self.picam.capture_array()
#             # print("Frame capture time:", (time.time() - t1) * 1000)
#
#             wall_mask = self.wall_detector.detect(frame)
#             color_frame, _ = self.color_detector.detect(frame.copy(), wall_mask)
#             victim_frame, cropped_vic, box = self.victim_cropper.crop(color_frame, wall_mask)
#             warped_vic = self.victim_cropper.Warp(cropped_vic)
#
#             if warped_vic is not None:
#                 best_match, _, similarity = self.victim_detector.detect(warped_vic)
#                 if best_match and box is not None:
#                     cv.drawContours(victim_frame, [box], 0, (255, 0, 0), 1)
#                     cv.putText(victim_frame, f"{similarity:.2f}",
#                                (box[0][0], box[0][1] - 35),
#                                cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#                     if similarity >= 0.95:
#                         victim_sender.FoundVictim('right', best_match)
#                         cv.putText(victim_frame, best_match,
#                                    (box[0][0], box[0][1] - 10),
#                                    cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
#
#             frame_count += 1
#             current_time = time.time()
#             elapsed = current_time - prev_time
#             if elapsed >= 1.0:
#                 self.fps = frame_count / elapsed
#                 frame_count = 0
#                 prev_time = current_time
#
#             cv.putText(victim_frame, f"FPS: {int(self.fps)}", (10, 30),
#                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
#             cv.imshow('Detection', victim_frame)
#             if cropped_vic is not None:
#                 cv.imshow('cropped', cropped_vic)
#             else:
#                 if cv.getWindowProperty('cropped', cv.WND_PROP_VISIBLE) >= 1:
#                     cv.destroyWindow('cropped')
#
#             if warped_vic is not None:
#                 cv.imshow('warped', warped_vic)
#             else:
#                 if cv.getWindowProperty('warped', cv.WND_PROP_VISIBLE) >= 1:
#                     cv.destroyWindow('warped')
#
#             if cv.waitKey(1) & 0xFF == ord('q'):
#                 self.running = False
#
#             t2 = time.time()
#             # print("Time elapsed: ", (t2 - t1) * 1000, "FPS: ", self.fps)
#
#     def stop(self):
#         self.running = False

class VisionProcess:
    """
    @param side 'R', 'L'
    """

    def __init__(self, side, width=160, height=120, fps=120):
        super().__init__()
        self.running = False
        self.fps = 0

        self.side = side

        self.wall_detector = WhiteWallDetector()
        self.color_detector = ColorDetector()
        self.victim_cropper = VictimCropper()
        self.victim_detector = VictimDetector()

        cam_id = get_camera_map()[self.side]
        self.picam = Picamera2(cam_id)
        self.picam_config = self.picam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            buffer_count=8,
        )
        self.picam_config["controls"]["FrameRate"] = fps  # Set FPS
        self.picam.configure(self.picam_config)
        self.picam.set_controls({
            # "AeEnable": False,
            "ExposureTime": 20000,  # 20 ms
            "NoiseReductionMode": 1,  # Fast. 0: Off, 1: Fast, 2: HighQuality, 3: Minimal, 4. ZeroShutterLag
            # "AnalogueGain": 1.0}
        })

        self.picam.start()

    def run(self):
        self.running = True
        prev_time = time.time()
        frame_count = 0
        while self.running:
            try:
                t1 = time.time()

                frame = self.picam.capture_array('main')
                wall_mask = self.wall_detector.detect(frame)
                color_frame, _ = self.color_detector.detect(frame.copy(), wall_mask)
                victim_frame, cropped_vic, box = self.victim_cropper.crop(color_frame, wall_mask)
                warped_vic = self.victim_cropper.Warp(cropped_vic)

                if warped_vic is not None:
                    best_match, _, similarity = self.victim_detector.detect(warped_vic)
                    if best_match and box is not None:
                        cv.drawContours(victim_frame, [box], 0, (255, 0, 0), 1)
                        cv.putText(victim_frame, f"{similarity:.2f}",
                                   (box[0][0], box[0][1] - 35),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        if similarity >= 0.95:
                            victim_sender.FoundVictim(self.side, best_match)
                            cv.putText(victim_frame, best_match,
                                       (box[0][0], box[0][1] - 10),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

                frame_count += 1
                current_time = time.time()
                elapsed = current_time - prev_time
                if elapsed >= 1.0:
                    self.fps = frame_count / elapsed
                    frame_count = 0
                    prev_time = current_time

                cv.putText(victim_frame, f"FPS: {int(self.fps)}", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if use_imshow:
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

                t2 = time.time()
                # print("Time elapsed: ", (t2 - t1) * 1000, "FPS: ", self.fps)
            except KeyboardInterrupt:
                self.running = False

    def stop(self):
        self.running = False


def process_vision(side: str):
    vision = VisionProcess(side, 160, 120, 200)
    vision.run()


if __name__ == "__main__":
    CAMERA_WIDTH = 160
    CAMERA_HEIGHT = 120
    TARGET_FPS = 200

    # cap_thread = VideoCaptureThread(
    #     src=0,
    #     width=CAMERA_WIDTH,
    #     height=CAMERA_HEIGHT,
    #     fps=TARGET_FPS
    # )

    # proc_thread = ProcessingThread(None)

    try:
        right_vision = mp.Process(target=lambda: process_vision('R'))
        left_vision = mp.Process(target=lambda: process_vision('L'))
        right_vision.start()
        left_vision.start()
        right_vision.join()
        left_vision.join()
        # right_vision = VisionProcess("R", width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=TARGET_FPS)
        # left_vision = VisionProcess("R", width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=TARGET_FPS)

        # right_vision.start()
        # left_vision.start()

        # right_vision.join()
        # left_vision.join()

    except KeyboardInterrupt:
        pass
    finally:
        # proc_thread.stop()
        # cap_thread.stop()
        # proc_thread.join()
        # cap_thread.join()
        cv.destroyAllWindows()
