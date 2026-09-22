# Dashcam Multiclass Object Detection

Upload a dashcam video, pick a trained checkpoint, and get back the video with detected `biker`/`car`/`pedestrian`/`truck`/`trafficLight` boxes drawn on every frame.

Deployed on Streamlit Community Cloud — the model, loss functions, and inference pipeline are the same code from the [driving_detection](https://github.com/thillaiambalam15/dashcam-multiclass-detection) project's `capstone_python_files/`, adapted here into a batch upload → process → result flow (no live stream).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
