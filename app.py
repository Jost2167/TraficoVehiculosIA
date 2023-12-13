import cv2
import pandas as pd
import numpy as np
import psycopg2
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import json
import time

from ultralytics import YOLO
from tracker import Tracker

app = Flask(__name__)

model = YOLO('modelov8.pt')

videoo = None
cap = None
video_paused = False  # Variable global para controlar el estado de pausa

paused = False
last_frame = None

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    global videoo, cap, paused, last_frame  # Acceder a las variables globales

    if request.method == 'POST':
        # Obtener el archivo de vídeo enviado por el usuario
        video_file = request.files['video']

        # Guardar el archivo en el servidor
        video_path = 'static/videos/' + video_file.filename
        video_file.save(video_path)

        videoo = video_path  # Asignar el nombre del video a la variable global
        cap = cv2.VideoCapture(videoo)  # Crear el objeto de captura de video

        # Restablecer la variable de control a su estado inicial
        paused = False
        last_frame = None

        # Redireccionar al reproductor de vídeo con el nombre del archivo como parámetro
        return redirect(url_for('play_video', filename=video_file.filename))

    return render_template('index.html')

my_file = open("clases.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

tracker = Tracker()

offset = 6
vh_counter = {}
counter = set()

line_points = []

# Configuración de la conexión a la base de datos
db_host = 'localhost'
db_port = '5432'
db_name = 'Traffic'
db_user = 'postgres'
db_password = 'Jhoncito01062004.'

# Función para establecer la conexión a la base de datos
def create_connection():
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        print('Conexión exitosa a la base de datos PostgreSQL')
        return conn
    except psycopg2.Error as e:
        print('Error al conectarse a la base de datos PostgreSQL:', e)
        return None

# Función para crear la tabla traffictable en la base de datos
def create_traffic_table():
    conn = create_connection()
    if conn is None:
        return

    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffictable (
                video_name VARCHAR(100),
                car_count INTEGER,
                video_duration INTEGER
            )
        ''')
        conn.commit()
        print('Tabla traffictable creada correctamente')
    except psycopg2.Error as e:
        print('Error al crear la tabla traffictable:', e)
    finally:
        cursor.close()
        conn.close()

def generate_frames():
    global count, video_paused
    counter = set()
    crossed_ids = set()
    count_once = 0
    last_car_count = 0

    while True:
        if video_paused:
            time.sleep(0.1)
            continue

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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        bbox_id = tracker.update(obj_list)

        if len(line_points) > 1:
            x1, y1 = line_points[0]
            x2, y2 = line_points[1]

            for bbox in bbox_id:
                x3, y3, x4, y4, id = bbox
                cx = int((x3 + x4) // 2)
                cy = int((y3 + y4) // 2)

                if (
                    ((cy > y1) and (cy < y2)) or ((cy > y2) and (cy < y1))
                ) and (
                    ((cx > x1) and (cx < x2)) or ((cx > x2) and (cx < x1))
                ):
                    if id not in counter:
                        counter.add(id)
                        crossed_ids.add(id)

            counter = {id for id in counter if id not in crossed_ids}

        if len(line_points) > 1:
            cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 2)
            cv2.putText(frame, 'Count line', (line_points[0][0], line_points[0][1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

        if len(line_points) > 1:
            count_once = len(crossed_ids)
            cv2.putText(frame, 'Count N. Vehicles: ' + str(count_once), (20, 70), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

        # Insertar datos en la base de datos si ha ocurrido un cambio en el recuento de automóviles
        if count_once != last_car_count:
            last_car_count = count_once

            video_name = videoo
            car_count = count_once
            video_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))

            conn = create_connection()
            if conn is not None:
                try:
                    cursor = conn.cursor()
                    cursor.execute('INSERT INTO traffictable (video_name, car_count, video_duration) VALUES (%s, %s, %s)', (video_name, car_count, video_duration))
                    conn.commit()
                except psycopg2.Error as e:
                    print('Error al insertar los datos en la tabla traffictable:', e)
                finally:
                    cursor.close()
                    conn.close()

@app.route('/pause', methods=['GET'])
def pause():
    global video_paused
    video_paused = True
    return ('', 204)  # Respuesta exitosa sin contenido

@app.route('/resume', methods=['GET'])
def resume():
    global video_paused
    video_paused = False
    return ('', 204)  # Respuesta exitosa sin contenido

@app.route('/play/<filename>')
def play_video(filename):
    return render_template('play.html', filename=filename)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_line', methods=['POST'])
def set_line():
    line_data = request.get_json()
    x1 = int(line_data['x1'])
    y1 = int(line_data['y1'])
    x2 = int(line_data['x2'])
    y2 = int(line_data['y2'])
    line_points.clear()
    line_points.append((x1, y1))
    line_points.append((x2, y2))

    response_data = {'message': 'Line coordinates received'}
    return jsonify(response_data)

@app.route('/count_updates')
def count_updates():
    def generate_count_updates():
        global count
        while True:
            yield f"data: {count}\n\n"
            count += 1
            time.sleep(1)

    return Response(generate_count_updates(), mimetype='text/event-stream')

if __name__ == '__main__':
    create_traffic_table()
    app.run(debug=True)

