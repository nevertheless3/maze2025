from picamera2 import Picamera2 

picam = Picamera2(0)

picam_config = picam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam.video_configuration.controls.FrameRate = 45.0

picam.configure(picam_config)
picam.set_controls({'ExposureTime': 20000})

picam.start()

while True:
    frame = picam.capture_array()