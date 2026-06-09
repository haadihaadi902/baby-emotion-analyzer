import cv2
from moviepy import VideoFileClip


def extract_audio_from_video(video_path, output_audio_path):
    video = VideoFileClip(video_path)

    if video.audio is None:
        video.close()
        raise ValueError("No audio found in uploaded video.")

    video.audio.write_audiofile(
        output_audio_path,
        logger=None
    )

    video.close()

    return output_audio_path


def extract_frame_from_video(video_path, output_image_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    middle_frame = total_frames // 2

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        middle_frame
    )

    success, frame = cap.read()

    if not success:
        cap.release()
        raise ValueError(
            "Could not extract frame."
        )

    cv2.imwrite(
        output_image_path,
        frame
    )

    cap.release()

    return output_image_path