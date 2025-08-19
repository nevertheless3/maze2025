import numpy as np 
# import serial
import time

# ser = serial.Serial('/dev/ttyS0', baudrate=115200, timeout=1)

class SendVictim:
    def __init__(self):
        self.victims_detections = {}
        self.sent_victims = set()
        self.reset_time = time.time()

    def send(self, victim_type):
        # ser.write(f"{victim_type}\n".encode('utf-8'))
        print(f"Sent: {victim_type}")

    def reset(self ,force_reset: bool = False, just_pins: bool = False):
        if force_reset or time.time() - self.reset_time > 500:
            if not just_pins:
                self.victims_detections = {}
                self.reset_time = time.time()


    def FoundVictim(self , letter):

        victim_checking_count = 3
        if letter in self.victims_detections:
            self.victims_detections[letter] += 1
        else:
            self.victims_detections[letter] = 1

        print("Detected victim", letter, "for", self.victims_detections[letter], "time(s)")

        if letter in self.victims_detections and sum(self.victims_detections.values()) >= victim_checking_count:
            best_letter = list(self.victims_detections.keys())[np.argmax(np.array(list(self.victims_detections.values())))]
            self.send(best_letter)
            print("----------------------Detected victim", best_letter, "after detecting for", self.victims_detections[letter], "time(s)")

            self.victims_detections[best_letter] = 0
            time.sleep(0.001)
            self.reset(True)
            self.reset_time = time.time()