from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cv2
import threading

app = Flask(__name__)

model = YOLO("wepbest.pt")

camera = None
camera_lock = threading.Lock()

# ---------------- DASHBOARD ----------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

# ---------------- START CAMERA ----------------
@app.route("/start_camera")
def start_camera():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
    return jsonify({"status": "camera started"})

# ---------------- STOP CAMERA ----------------
@app.route("/stop_camera")
def stop_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    return jsonify({"status": "camera stopped"})

# ---------------- VIDEO STREAM ----------------
def gen_frames():
    global camera
    while True:
        with camera_lock:
            if camera is None:
                break
            success, frame = camera.read()

        if not success:
            break

        results = model(frame, conf=0.45)

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if int(box.cls[0]) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"Weapon {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(debug=True)
