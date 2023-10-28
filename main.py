import os
import cv2
import time
import datetime
from loguru import logger
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from lib.detector import YOLOv8
from lib.grab_screen import grab_screen
from lib.sharedmemory import SharedMemory
    

dataset_path = os.path.join(os.getcwd(), "ETSMotion")
yolov8_path = 'weights/yolov8n.onnx'

telemetrymem = SharedMemory()
yolov8 = YOLOv8(yolov8_path)

data = telemetrymem.update()
source = data.jobCitySource.decode('utf-8').strip(b'\x00'.decode())
destination = data.jobCityDestination.decode('utf-8').strip(b'\x00'.decode())
assert source != '' and destination != '', "scs-telemetry didn't work!"
logger.info("source: ", source)
logger.info("destination: ", destination)

d = datetime.datetime.now()
folder_name = str(source) + '-' + str(destination) + '-' + str(d.year) + '-' + str(d.month) + '-' + str(d.day) + '\\'
folder_path = os.path.join(dataset_path, folder_name)
os.makedirs(folder_path)

fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
fps = 20
lanevideo = cv2.VideoWriter('lane.mp4', fourcc, fps, (480, 120))
navvideo = cv2.VideoWriter('nav.mp4', fourcc, fps, (64, 64))

img = grab_screen()
img = cv2.resize(img, (1280, 720))
tl_area = img[0:300, 390:890, :]
control_data = []

while True:
    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=2)
    thread1 = pool.submit(grab_screen)
    thread2 = pool.submit(yolov8.infer, tl_area)
    
    img = thread1.result()
    stop = thread2.result()
    print(stop)
    
    nav = img[610:710, 630:730, :]
    nav = cv2.resize(nav, (64, 64))
    
    img = cv2.resize(img, (1280, 720))
    lane_img = img[330:570, 160:1120, :]
    lane_img = cv2.resize(lane_img, (480, 120))
    
    tl_area = img[0:300, 390:890, :]
    
    data = telemetrymem.update()
    speed = data.speed
    speedlimit = data.speedlimit
    steering = data.gameSteer
    throttle = data.gameThrottle
    brake = data.userBrake
    
    if throttle != 0:
        power = throttle
    else:
        power = - brake
    iter_data = [speed, speedlimit, steering, power]
    control_data.append(iter_data)
    panel = np.zeros((160, 300, 3))
    panel = cv2.putText(panel, "speed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    panel = cv2.putText(panel, "speedlimit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    panel = cv2.putText(panel, "steering", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    panel = cv2.putText(panel, "throttle", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    panel = cv2.putText(panel, "brake", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    panel = cv2.putText(panel, str(speed), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    panel = cv2.putText(panel, str(speedlimit), (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    panel = cv2.putText(panel, str(steering), (150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    panel = cv2.putText(panel, str(throttle), (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    panel = cv2.putText(panel, str(brake), (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # cv2.imshow('ori', img)
    #cv2.imshow('lane', lane_img)
    #cv2.imshow('nav', nav)
    #cv2.imshow('tl', tl_area)
    cv2.imshow('panel', panel)
    
    lanevideo.write(lane_img)
    navvideo.write(nav)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    t1 = time.time()
    iter_time = t1 -  t0
    if iter_time < 0.05:
        time.sleep(0.05 - iter_time)
    
control_data = np.array(control_data)
np.save('', control_data)

lanevideo.release()
navvideo.release()
cv2.destroyAllWindows()