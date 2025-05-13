import cv2
import dlib
import imutils
from capture import *

while True:
     ret, frame = cap.read()
     if ret == False:
          break
     flipped_frame = cv2.flip(frame, 1)
     frame = imutils.resize(flipped_frame, width=1080)
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     coordinates_bboxes = detector(gray, 1)
     #print("coordinates_bboxes:", coordinates_bboxes)
     for c in coordinates_bboxes:
          x_ini, y_ini, x_fin, y_fin = c.left(), c.top(), c.right(), c.bottom()
          cv2.rectangle(frame, (x_ini, y_ini), (x_fin, y_fin), (0, 255, 0), 1)
          shape = predictor(gray, c)
          for i in range(0, 68):
               x, y = shape.part(i).x, shape.part(i).y
               cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
               cv2.putText(frame, str(i + 1), (x, y -5), 1, 0.8, (0, 255, 255), 1)
     cv2.imshow("Frame", frame)
     if cv2.waitKey(1) & 0xFF == 13:
          break
cap.release()
cv2.destroyAllWindows()