import cv2
import numpy as np
SHAPE = 600
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
result = cv2.VideoWriter('trial_simple.avi', fourcc, 20.0, (SHAPE, SHAPE), False) # better to use os.join here

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (SHAPE, SHAPE)) # Very important
    result.write(gray)
    cv2.imshow("Frame", gray)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
result.release()

cv2.destroyAllWindows()