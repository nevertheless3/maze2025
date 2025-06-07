import sensor, time, image, math, utime, time, pyb
from ulab import numpy as np
from pyb import LED, millis
from machine import Pin

green_led = pyb.LED(2)
red_led = pyb.LED(1)
blue_led = pyb.LED(3)

# MUST BE CHANGED///////////////////////////////////////////////////////
PIN3 = Pin('P3', Pin.OUT)
PIN4 = Pin('P4', Pin.OUT)
PIN5 = Pin('P5', Pin.OUT)
PIN6 = Pin('P6', Pin.OUT)

RED_THRESH = (0, 80, 18, 60, -15, 40)
GREEN_THRESH = (35, 80, -35, -9, 7, 41)
YELLOW_THRESH = (0, 80, -15, 17, 23, 57)
THRESHOLD = [RED_THRESH, GREEN_THRESH, YELLOW_THRESH]
COLORS = [0, (255, 0, 0), (0, 255, 0), 0, (255, 215, 0), 0, 0, 0, 0]
LIGHTNING = (90, 100, -128, 127, -128, 127)

timer1, counter, u_count , h_count , s_count , y_count , g_count , r_count = 0, 0 , 0 , 0 ,0 , 0 , 0 ,0
side = 'right'

RED = 1
GREEN = 2
BLUE = 3
MAGENTA = 4
CYAN = 5
YELLOW = 6

victims_detections = {}
victims_pins = {
    'H': ((0, 0, 1, 0), MAGENTA),
    'S': ((0, 1, 0, 0), BLUE),
    'U': ((0, 1, 1, 0), CYAN),
    'red': ((1, 0, 0, 0), RED),
    'yellow': ((1, 0, 1, 0), YELLOW),
    'green': ((1, 1, 0, 0), GREEN)
}


class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA)
sensor.skip_frames(time=100)

patterns = {
    'H': [
        np.array([[1, 0, 1],
                  [1, 1, 1],
                  [1, 0, 1]]),

        np.array([[1, 1, 1],
                  [0, 1, 0],
                  [1, 1, 1]])
    ],

    'S': [

        np.array([[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]]),

        np.array([[0, 1, 1],
                  [1, 1, 1],
                  [1, 1, 0]]),

        np.array([[1, 1, 0],
                  [1, 1, 1],
                  [0, 1, 1]]),

        np.array([[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 0]]),

        np.array([[0, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]]),

        np.array([[0, 1, 1],
                  [1, 1, 0],
                  [1, 1, 0]]),

        np.array([[1, 1, 1],
                  [1, 1, 1],
                  [0, 1, 1]])
    ],

    'U': [
        np.array([[1, 0, 1],
                  [1, 0, 1],
                  [1, 1, 1]]),

        np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 0, 1]]),

        np.array([[1, 1, 1],
                  [0, 0, 1],
                  [1, 1, 1]]),

        np.array([[1, 1, 1],
                  [1, 0, 0],
                  [1, 1, 1]])
    ]
}


clock = time.clock()

def argmax (l):
    a=0
    for i in range(len(l)):
        if l[i]>l[a]:
            a=i
    return a

def WheelCropped(img_width):
    global side
    if side == 'left':
        wheel_x = 58 #right
        frame_width = img_width - wheel_x #right
    elif side == 'right':
        wheel_x = 120 #left
        frame_width = wheel_x #left

    return wheel_x , frame_width

def set_pins(pins: tuple):
    if len(pins) != 4:
        return
    pins_items = [PIN3, PIN4, PIN5, PIN6]
    for idx, pin in enumerate(pins):
        if pin:
            pins_items[idx].high()
        else:
            pins_items[idx].low()


def blink(color):
    led = None
    led2 = None
    if color == RED:
        led = red_led

    elif color == GREEN:
        led = green_led

    elif color == BLUE:
        led = blue_led

    elif color == MAGENTA:
        led = blue_led
        led2 = red_led

    elif color == YELLOW:
        led = red_led
        led2 = green_led

    elif color == CYAN:
        led = green_led
        led2 = blue_led

    led.on()
    if led2 is not None:
        led2.on()
    time.sleep(0.2)
    led.off()
    if led2 is not None:
        led2.off()


def AreaConditions(blob, wheel_x):
    global side
    # print(blob.pixels() , blob.area() , blob.w() , blob.pixels() / blob.area() , blob.w() / blob.h() , blob.h() / blob.w() , blob.x(), blob.x() + blob.w(),blob.y())
    if side == 'left':
        return (
                5000 >= blob.pixels() >= 100 and
                8500 >= blob.area() >= 400 and
                10 <= blob.w() <= 90 and
                0.1 <= blob.pixels() / blob.area() <= 0.8 and
                0.5 <= blob.w() / blob.h() <= 2 and
                0.5 <= blob.h() / blob.w() <= 2 and
                wheel_x + 2 <= blob.x() and
                blob.w() < 70 and
                blob.x() + blob.w() <= 158 and
                8 <= blob.y()
                # blob.cx() <120
        )

    elif side == 'right':
        return (
                5000 >= blob.pixels() >= 100 and
                8500 >= blob.area() >= 400 and
                10 <= blob.w() <= 90 and
                0.1 <= blob.pixels() / blob.area() <= 0.8 and
                0.5 <= blob.w() / blob.h() <= 2 and
                0.5 <= blob.h() / blob.w() <= 2 and
                2 <= blob.x() and
                blob.w() < 70 and
                blob.x() + blob.w() <= wheel_x-2 and
                8 <= blob.y()
                # blob.cx() > 45
        )

def LittleAreaConditions(blob):
    return (
            0.5 <= blob.w() / blob.h() <= 2 and
            0.5 <= blob.h() / blob.w() <= 2 and
            blob.pixels() >= 100
    )


def ColorBlobsConditions(blob):
    width = blob.w()
    height = blob.h()
    blob_area = blob.w() * blob.h()

    return (
            0.75 <= width / (height + 10 ** -6) <= 1.5 and
            300 < blob_area < 5000
    )


reset_time = 0
def reset(force_reset: bool = False, just_pins: bool = False):
    global victims_detections
    global reset_time
    if force_reset or millis() - reset_time > 500:
        PIN3.low()
        PIN4.low()
        PIN5.low()
        PIN6.low()
        if not just_pins:
            victims_detections = {}
            reset_time = millis()
        # print("Reseted")



def filter_color(blob):
    # print('----------' ,blob.pixels() ,  blob.area() / 19200)
    aspect_ratio = blob.w() / blob.h()
    if not (0.25 < aspect_ratio < 1.75):
        # print('hello')
        return False

    if blob.area() > 9000:
        # print('oooiiii')
        return False
    if blob.pixels() > 9000:
        # print('iiiiiioooo')
        return False
    if blob.pixels() < 100 or blob.area() < 100:
        return False

    else:
        return True


def FoundVictim(letter):
    global timer1, counter, reset_time
    global victims_detections

    victim_checking_count = 3
    if letter in victims_detections:
        victims_detections[letter] += 1
    else:
        victims_detections[letter] = 1

    # ("Detected victim", letter, "for", victims_detections[letter], "time(s)")

    if letter in victims_detections and sum(victims_detections.values()) >= victim_checking_count:
        best_letter = list(victims_detections.keys())[argmax(np.array(list(victims_detections.values())))]
        set_pins(victims_pins[best_letter][0])
        blink(victims_pins[best_letter][1])

        # print("----------------------Detected victim", best_letter, "after detecting for", victims_detections[letter], "time(s)")

        # if victims_detections[best_letter] >= victim_checking_count:
        victims_detections[best_letter] = 0
        time.sleep(0.001)
        reset(True)
        reset_time = millis()


    # elif letter not in victims_detections:
    #



def CheckBlob(blob, THRESH, code):
    block_width = blob.width() // 4
    block_height = blob.height() // 4
    avg_color = None
    count = 0

    if code == 4:
        thresh = THRESHOLD[2]
    elif code == 1:
        thresh = THRESHOLD[0]
    elif code == 2:
        thresh = THRESHOLD[1]

    for i in range(4):
        for j in range(4):
            try:
                block = blob.copy(roi=(j * block_width, i * block_height, block_width, block_height))
            except:
                break
            avg_color = block.get_statistics()
            l_avg = avg_color.l_mean()
            a_avg = avg_color.a_mean()
            b_avg = avg_color.b_mean()

            try:
                if (thresh[0] <= l_avg <= thresh[1]) and \
                        (thresh[2] <= a_avg <= thresh[3]) and \
                        (thresh[4] <= b_avg <= thresh[5]):
                    count += 1
            except:
                None

    if count >= 7:
        print(count)
        return True
    else:
        print(count)
        return False

def CropWall(img_height , img_width , wheel_x , frame_width):
    wall = img_height
    for y in range(img_height-1, 1 , -1):
        color = img.get_pixel(frame_width//3 , y)
        color_check = img.get_pixel((frame_width//3)*2 , y)

        if ((abs(color[0] - color[1]) < 20) and \
           (abs(color[1] - color[2]) < 20) and \
           (abs(color[2] - color[0]) < 20) and \
           color[0] > 80 and color[1] > 80 and color[2] > 80) \
           or  ((abs(color_check[0] - color_check[1]) < 20) and \
          (abs(color_check[1] - color_check[2]) < 20) and \
          (abs(color_check[2] - color_check[0]) < 20) and \
          color_check[0] > 50 and color_check[1] > 50 and color_check[2] > 50):

            wall = y
            if wall -20 >= 0:
                noise_check = img.get_pixel((frame_width//3) , wall - 20)
                noise_check1 = img.get_pixel((frame_width//3)*2 , wall - 20)

                if noise_check is not None :
                    if not(125 <= noise_check[0] <= 255 and 125<= noise_check[1] <= 255 and 125 <= noise_check[2]<= 255) and \
                     not(125 <= noise_check1[0] <= 255 and 125<= noise_check1[1] <= 255 and 125 <= noise_check1[2]<= 255):
                        continue
            break

    return wall


while True:
    try:
        clock.tick()
        sensor.set_pixformat(sensor.RGB565)
        img = sensor.snapshot()
        img_height = img.height()
        img_width = img.width()
        wheel_x , frame_width = WheelCropped(img_width)
        wall = CropWall(img_height , img_width , wheel_x , frame_width)

        if side == 'left':
            wall_x_left_side = frame_width
            wall_x_right_side = frame_width

            for x in range(wheel_x , frame_width):
                color = img.get_pixel(x , wall)
                if color is not None:
                    if ((abs(color[0] - color[1]) < 20) and \
                       (abs(color[1] - color[2]) < 20) and \
                       (abs(color[2] - color[0]) < 20) and \
                       color[0] > 80 and color[1] > 80 and color[2] > 80):
                        wall_x_left_side  = x
                        break




            for x2 in range(img_width-1 , 1 , -1):
                color = img.get_pixel(x2 , wall)
                if color is not None:
                    if ((abs(color[0] - color[1]) < 20) and \
                       (abs(color[1] - color[2]) < 20) and \
                       (abs(color[2] - color[0]) < 20) and \
                       color[0] > 80 and color[1] > 80 and color[2] > 80):
                        if wall_x_right_side >= wheel_x:
                            wall_x_right_side = x2
                            break


        elif side == 'right':
            wall_x_left_side = 0
            wall_x_right_side = wheel_x

            for x in range(wheel_x):
                color = img.get_pixel(x , wall)
                if color is not None:
                    if ((abs(color[0] - color[1]) < 20) and \
                       (abs(color[1] - color[2]) < 20) and \
                       (abs(color[2] - color[0]) < 20) and \
                       color[0] > 80 and color[1] > 80 and color[2] > 80):
                        wall_x_left_side  = x
                        break




            for x2 in range(wheel_x-1 , 1 , -1):
                color = img.get_pixel(x2 , wall)
                if color is not None:
                    if ((abs(color[0] - color[1]) < 20) and \
                       (abs(color[1] - color[2]) < 20) and \
                       (abs(color[2] - color[0]) < 20) and \
                       color[0] > 80 and color[1] > 80 and color[2] > 80):
                        if wall_x_right_side >= wheel_x:
                            wall_x_right_side = x2
                            # print(wall_x_right_side)
                            break




        region = {'left': (wall_x_left_side , 0 , abs(wall_x_right_side - wheel_x), wall-2),
                    'right':(wall_x_left_side, 0 , wall_x_right_side - wall_x_left_side, wall-2)}
        region_of_interest = region[side]

        # img.draw_rectangle(region_of_interest ,(255  , 255 , 255) , 3 )
        # img_copy = img.copy()
        is_color = False
        try:
            blbs = img.find_blobs(THRESHOLD, area_threshold=400, pixels_threshold=100, roi= region_of_interest , merge=True)
        except Exception:
            pass
            # print(e , region_of_interest)



        victim_detected = False
        if len(blbs) > 0:
            # print("found clr")
            for b in blbs:
                # print(filter_color(b))
                if filter_color(b) != False:
                    try:
                        blob_img = img.copy(roi=(b.x(), b.y(), b.w(), b.h()))
                    except:
                        continue
                    if CheckBlob(blob_img, THRESHOLD, b.code()):
                        img.draw_rectangle(b.x(), b.y(), b.w(), b.h(), COLORS[b.code()], 3, False)
                        img.draw_cross(b.cx(), b.cy())

                        if b.code() == 1:
                            is_color = True
                            FoundVictim('red')
                            print("red")
                        elif b.code() == 2:
                            is_color = True
                            FoundVictim('green')
                            print("green")
                        elif b.code() == 4:
                            is_color = True
                            FoundVictim('yellow')
                            print("yellow", b.area() , b.pixels())

        if is_color == False:
            is_blob = False
            lines = []
            img.gamma_corr(gamma=2, contrast=4, brightness=-2)
            img.to_grayscale()
            img.gaussian(2)
            img.mean(5, threshold=True, offset=16, invert=True)
            img.lens_corr(strenght=1.8)
            # img.median(2, percentile=0.75,offset=2, invert=True)
            try:
                vic1 = img.find_blobs([(200, 255)], roi= region_of_interest)
            except Exception:
                pass
                # print(e, region_of_interest)
            for vic in vic1:
                # print(AreaConditions(vic, wheel_x))
                # img.draw_rectangle(vic.rect() , (255 , 255 , 255) , 1)
                if AreaConditions(vic, wheel_x):

                    # img.draw_rectangle(vic.rect() , (255 , 255 , 255) , 1)
                    is_blob = True
                    try:
                        img =img.crop(roi = (vic.x()-int(vic.w()/10),vic.y()-int(vic.h()/10),vic.w()+int(vic.w()/5),vic.h()+int(vic.h()/5)))
                    except:
                        continue

                    lines = []
                    lines_dict = {}
                    # for l in img.find_lines(theta_margin=10, rho_margin=20):
                    # for l in img.find_lines(theta_margin=10, rho_margin=40):
                    for l in img.find_lines(threshold=3500 , theta_margin=30):

                    # for l in img.find_lines(threshold=2500, theta_margin=30, rho_margin=20):
                        theta = l.theta()
                        if theta in lines_dict:
                            lines_dict[theta] += 1
                        else:
                            lines_dict[theta] = 1
                        lines.append(theta)

                    if len(lines) == 0:
                        continue

                    final_theta = list(lines_dict.keys())[np.argmax(list(lines_dict.values()))]
                    final_theta %= 90;

                    if final_theta == 0:
                        continue

                    # print(lines)
                    try:
                        img = img.rotation_corr(0, 0, -final_theta)

                        second_blb = img.find_blobs([(200 , 255)])
                        for b in second_blb:
                            if LittleAreaConditions(b):
                                # print('----------',b.rect())
                                img = img.crop(roi = (b.rect()))

                    except Exception:
                        continue

            HSU_MAT_X = 3
            HSU_MAT_Y = 3

            if is_blob:
                full_black = False
                block_width = img.width() // HSU_MAT_Y
                block_height = img.height() // HSU_MAT_X

                # block_widths_s = img_copy.width() // 5
                # block_heights_s = img_copy.height() // 5

                binary_matrix = np.zeros((HSU_MAT_X, HSU_MAT_Y), dtype=np.uint8)
                timer1 = pyb.millis()
                count_avg = 0
                block_widths = block_width  # , block_widths_s]
                block_heights = block_height  # , block_heights_s]
                for i in range(HSU_MAT_X):
                    for j in range(HSU_MAT_Y):
                        try:
                            block = img.copy(roi=(j * block_width, i * block_height, block_width, block_height))
                        except:
                            continue
                        stats = block.get_statistics()
                        avg_color = stats.mean()
                        avg_l = stats.l_mean()
                        # print(count_avg , avg_color, avg_l)
                        if avg_l > 95:
                            count_avg += 1

                        if count_avg > 1:
                            full_black = True

                        binary_matrix[i, j] = 1 if avg_color > 35 else 0
                        # print(avg_color)
                print(binary_matrix)
                for letter, patterns_list in patterns.items():

                    for pattern in patterns_list:
                        if pattern.shape == binary_matrix.shape:
                            if np.all(pattern == binary_matrix):
                                FoundVictim(letter)
                                print(style.RED ,'-------' ,letter , '-------',style.RESET)
                                # print(binary_matrix)

        reset()
        reset(True, True)
        # print(clock.fps())
    except:
        pass
