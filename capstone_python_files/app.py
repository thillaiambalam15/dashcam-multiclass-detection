import os, re
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time

import cv2
import torch
from flask import Flask, Response, request, redirect
from werkzeug.utils import secure_filename

from inference import load_model, preprocess_frame_gpu, decode_predictions, nms, draw_detections

HERE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(HERE, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov"}

VIDEO_PATH = None


CHECKPOINT_NAME = "modelA_iou"
current_checkpoint = CHECKPOINT_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

model = load_model(CHECKPOINT_NAME, False, device)

app = Flask(__name__)


def gen_frames():
    if not VIDEO_PATH:
        return

    placeholder = 30 * torch.ones((480, 640, 3), dtype=torch.uint8).numpy()
    cv2.putText(placeholder, "Loading first frame...", (30, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
    ok, buffer = cv2.imencode(".jpg", placeholder)
    if ok:
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()
        curr_time = time.time()
        fps_now = 1 / (curr_time - prev_time)
        prev_time = curr_time

        img_tensor, pad_size, pad_left, pad_top = preprocess_frame_gpu(frame, device)
        img_batch = img_tensor.unsqueeze(0)
        with torch.no_grad():
            raw_output = model(img_batch)

        box, predlabel, confidence = decode_predictions(raw_output)
        final_box, final_predlabel, final_confidence = nms(box, predlabel, confidence)

        frame = draw_detections(frame, final_box, final_predlabel, final_confidence,
                                 pad_size, pad_left, pad_top)
        cv2.putText(frame, f"FPS: {fps_now:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), thickness=2)

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        frame_bytes = buffer.tobytes()

        elapsed_ms = (time.time() - frame_start) * 1000
        remaining = max(0, (1000 / fps) - elapsed_ms) / 1000
        if remaining > 0:
            time.sleep(remaining)

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


@app.route("/upload", methods=["POST"])
def upload():
    global VIDEO_PATH

    file = request.files.get("video")
    if file and file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext in ALLOWED_EXTENSIONS:
            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_DIR, filename)
            file.save(save_path)
            VIDEO_PATH = save_path

    return redirect("/")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    global model, current_checkpoint

    selected_model = request.args.get("model", current_checkpoint)

    if selected_model != current_checkpoint:
        model = load_model(selected_model, False, device)
        current_checkpoint = selected_model

    status = f"Currently loaded: {current_checkpoint}"

    if VIDEO_PATH:
        video_status = f"Video loaded: {re.split(r'[\\/]', VIDEO_PATH)[-1]}"
    else:
        video_status = "No video uploaded yet"

    return f"""
    <html>
      <head><title>Driving detection live demo</title></head>

      <body style="background:#111; text-align:center;">

        <h2 style="color:white;">{status}</h2>
        <h2 style="color:white;">{video_status}</h2>

        <form method="get" action="/">
         <select name="model">
           <option value="modelA_iou">Model A (IoU loss)</option>
           <option value="modelA_eiou">Model A (EIoU loss)</option>
           <option value="modelB_iou">Model B (IoU loss)</option>
           <option value="modelB_eiou">Model B (EIoU loss)</option>
         </select>
         <button type="submit">Load model</button>
       </form>

       <form method="post" action="/upload" enctype="multipart/form-data">
           <input type="file" name="video" accept=".mp4,.avi,.mov">
           <button type="submit">Upload video</button>
       </form>

        <br><button onclick="startVideo()">Start processing</button>
        <br><br>
        <script>
            function startVideo(){{
                document.getElementById('feed').src = '/video_feed'}}
        </script>

        <img id="feed" style="max-width:90vw; max-height:85vh; width:auto; height:auto;">

      </body>
    </html>
    """


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
