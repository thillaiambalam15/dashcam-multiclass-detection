import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tempfile

import cv2
import imageio
import torch
import streamlit as st

from inference import load_model, preprocess_frame_gpu, decode_predictions, nms, draw_detections

CHECKPOINTS = {
    "ResNet with pretrained weights (IoU loss)": "modelA_iou",
    "ResNet with pretrained weights (EIoU loss)": "modelA_eiou",
    "ResNet trained from scratch (IoU loss)": "modelB_iou",
    "ResNet trained from scratch (EIoU loss)": "modelB_eiou",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def get_model(checkpoint_name):
    return load_model(checkpoint_name, False, device)


def process_video(video_path, checkpoint_name, progress_bar):
    model = get_model(checkpoint_name)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", format="FFMPEG")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_tensor, pad_size, pad_left, pad_top = preprocess_frame_gpu(frame, device)
        img_batch = img_tensor.unsqueeze(0)
        with torch.no_grad():
            raw_output = model(img_batch)

        box, predlabel, confidence = decode_predictions(raw_output)
        final_box, final_predlabel, final_confidence = nms(box, predlabel, confidence)
        frame = draw_detections(frame, final_box, final_predlabel, final_confidence,
                                 pad_size, pad_left, pad_top)

        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
        if total_frames > 0:
            progress_bar.progress(frame_idx / total_frames, text=f"Processing frame {frame_idx}/{total_frames}")

    cap.release()
    writer.close()
    return out_path


st.title("Dashcam Multiclass Object Detection")

st.caption(
    "This is a hosted demo, so it can't do real-time processing here, so it processes the full video "
    "and shows the result once done. For a real-time detection demo, check out the "
    "[GitHub repo](https://github.com/thillaiambalam15/dashcam-multiclass-detection) and run the "
    "included Flask app locally in your browser."
)

uploaded_file = st.file_uploader("Upload dashcam video", type=["mp4", "avi", "mov"])
checkpoint_label = st.selectbox("Model checkpoint", list(CHECKPOINTS.keys()))

if st.button("Process video"):
    if uploaded_file is None:
        st.warning("Upload a video first.")
    else:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_input:
            tmp_input.write(uploaded_file.read())
            input_path = tmp_input.name

        progress_bar = st.progress(0, text="Starting...")
        output_path = process_video(input_path, CHECKPOINTS[checkpoint_label], progress_bar)
        progress_bar.empty()

        st.video(output_path)
