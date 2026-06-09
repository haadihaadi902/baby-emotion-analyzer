import os
import uuid
import cv2

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model

from utils.audio_predict import predict_audio
from utils.image_predict import predict_image
from utils.baby_detector import verify_baby


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Load models
audio_model = load_model("models/baby_cry_cnn.keras")
image_model = load_model("models/young_affectnet_mobilenetv2_finetuned.keras")
baby_verify_model = load_model("models/baby_verify_model.keras")


def extract_audio_from_video(video_path, output_audio_path):
    video = VideoFileClip(video_path)

    if video.audio is None:
        video.close()
        raise ValueError("No audio found in uploaded video.")

    video.audio.write_audiofile(output_audio_path, logger=None)
    video.close()

    return output_audio_path


def extract_frame_from_video(video_path, output_image_path):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

    success, frame = cap.read()

    if not success:
        cap.release()
        raise ValueError("Could not extract frame from video.")

    cv2.imwrite(output_image_path, frame)
    cap.release()

    return output_image_path


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "video" not in request.files:
        return render_template(
            "error.html",
            message="No video file uploaded."
        )

    video_file = request.files["video"]

    if video_file.filename == "":
        return render_template(
            "error.html",
            message="No video selected."
        )

    unique_id = str(uuid.uuid4())
    filename = secure_filename(video_file.filename)

    video_path = os.path.join(UPLOAD_FOLDER, unique_id + "_" + filename)
    audio_path = os.path.join(UPLOAD_FOLDER, unique_id + "_audio.wav")
    frame_path = os.path.join(UPLOAD_FOLDER, unique_id + "_frame.jpg")

    try:
        video_file.save(video_path)

        # Extract frame first
        extract_frame_from_video(video_path, frame_path)

        # Baby verification
        is_baby = verify_baby(frame_path, baby_verify_model)

        if not is_baby:
            return render_template(
                "error.html",
                message="Please upload a baby crying video only."
            )

        # Extract audio
        extract_audio_from_video(video_path, audio_path)

        # Predict cry reason
        audio_result = predict_audio(audio_path, audio_model)

        # Predict emotion
        image_result = predict_image(frame_path, image_model)

        return render_template(
            "result.html",

            cry_reason=audio_result["cry_reason"].replace("_", " ").title(),
            cry_confidence=round(audio_result["cry_confidence"] * 100, 2),
            cry_intensity=audio_result["cry_intensity"].capitalize(),

            emotion=image_result["emotion"].capitalize(),
            emotion_confidence=round(image_result["emotion_confidence"] * 100, 2),
            emotion_intensity=image_result["emotion_intensity"].capitalize()
        )

    except Exception as e:
        return render_template(
            "error.html",
            message=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True, port=5001)