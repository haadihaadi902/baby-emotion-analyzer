import numpy as np
from tensorflow.keras.preprocessing import image


EMOTION_CLASSES = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


def get_intensity(score):
    """
    Convert confidence score into intensity level.
    """
    if score < 0.40:
        return "low"
    elif score < 0.70:
        return "medium"
    else:
        return "high"


def predict_image(image_path, image_model):
    """
    Predict emotion from an image/frame.
    """

    # Load image and resize to MobileNetV2 input size
    img = image.load_img(image_path, target_size=(160, 160))

    # Convert image to array
    img_array = image.img_to_array(img)

    # Normalize image
    img_array = img_array / 255.0

    # Add batch dimension: (1,160,160,3)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = image_model.predict(img_array, verbose=0)

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    emotion = EMOTION_CLASSES[predicted_index]
    intensity = get_intensity(confidence)

    return {
        "emotion": emotion,
        "emotion_confidence": round(confidence, 4),
        "emotion_intensity": intensity
    }