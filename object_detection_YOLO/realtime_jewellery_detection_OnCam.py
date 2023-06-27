
###### Real Time Jewelry Detection on Webcam ######
#--------------------------------------------------------------------------------------------------------------

import torch
import cv2
import numpy as np

path = 'E:/Task_CloudMantra/Object_detection/CUSTOM_OD/best.pt'

model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload= True)

capture = cv2.VideoCapture(0)

# Get the resolution of the camera feed
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_size = (width, height)

while True:
    ret, frame = capture.read()
    frame = cv2.resize(frame, output_size)
    results = model(frame)
    frame = np.squeeze(results.render())
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

capture.release()
cv2.destroyAllWindows()
