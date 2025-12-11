from picamera2 import Picamera2
import numpy as np
import cv2 as cv
import pickle

def read_file(path):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
        return obj

picam = Picamera2(1)
picam_config = picam.create_preview_configuration(
    main={"size": (640,480), "format": "RGB888"},
    buffer_count=8,
)
picam_config["controls"]["FrameRate"] = 60 
picam.configure(picam_config)
picam.set_controls({
    "ExposureTime": 20000,  
    "NoiseReductionMode": 1,  
})

picam.start()


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / (gamma + 10e-6)
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv.LUT(image, table)

def find_gamma(vs):
    x = (-2*vs + 400)/100
    return x if x > 1 else 1


cv.namedWindow("s")
cv.createTrackbar("gamma", "s" , 1, 500, lambda x: None)

mtx = read_file('distortion/mtx.pkl')
dist = read_file('distortion/dist.pkl')
newcameramtx = read_file('distortion/newcameramtx.pkl')
roi = read_file('distortion/roi.pkl')

while True:
    frame = picam.capture_array()

    #t=frame

    t = cv.undistort(frame, mtx, dist, None, newcameramtx)

    x,y,w,h = roi
    t=t[y:y+h, x:x+w]
    t=cv.resize(t, (frame.shape[1], frame.shape[0]))

    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_vs = np.mean(frame_hsv[:,:,2])

    #r = cv.getTrackbarPos("gamma","s")/100

    r = find_gamma(frame_vs)

    gamma_adjusted = adjust_gamma(frame, r)

    cv.putText(gamma_adjusted, str(frame_vs), (80,80), cv.FONT_HERSHEY_SIMPLEX,0.9, (255,200,150) ,2,cv.LINE_AA)

    cv.putText(gamma_adjusted, str(r), (80,100), cv.FONT_HERSHEY_SIMPLEX,0.9, (255,200,150) ,2,cv.LINE_AA)

    cv.imshow('f',gamma_adjusted)
    cv.imshow('w',t)

    # cv.imshow('ff', frame)

    if cv.waitKey(1) == 27:
        break

