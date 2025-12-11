import os
import time

while True:
    os.system("sudo pinctrl 17 op dl")

    time.sleep(1)

    os.system("sudo pinctrl 17 op dh")

    time.sleep(1)

