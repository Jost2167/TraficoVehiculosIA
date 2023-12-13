import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

model = YOLO('modelov8.pt')

mouse_points = []

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_points) < 2:  # Solo recogemos 2 puntos para una línea
            mouse_points.append((x, y))

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', mouse_click)

cap = cv2.VideoCapture('carros-1900.mp4')

my_file = open("clases.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

tracker = Tracker()

offset = 6
vh_counter = {}
counter = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)

    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")

    obj_list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            obj_list.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibuja un recuadro verde alrededor del coche detectado

    bbox_id = tracker.update(obj_list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int((x3 + x4) // 2)
        cy = int((y3 + y4) // 2)

        if len(mouse_points) > 1:
            x1, y1 = mouse_points[0]
            x2, y2 = mouse_points[1]

            # Calcula la distancia entre el punto medio del objeto y el segmento de línea
            dist = cv2.pointPolygonTest(np.array([(x1, y1), (x2, y2)]), (cx, cy), True)

            if abs(dist) <= offset and id not in vh_counter:
                vh_counter[id] = cy
                counter.add(id)

    if len(mouse_points) > 1:
        cv2.line(frame, mouse_points[0], mouse_points[1], (173, 255, 47), 2)
        cv2.putText(frame, ('Count line'), (mouse_points[0][0], mouse_points[0][1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255),2)
    d = len(counter)
    cv2.putText(frame, ('Count:') + str(d), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(30) & 0xFF == 27:  # Un delay de 30 ms dará una reproducción de video a aproximadamente la velocidad normal.
        break

cap.release()
cv2.destroyAllWindows()
