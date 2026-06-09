import librosa
import numpy as np

# Cry class labels
CRY_CLASSES = [
    "belly_pain",
    "burping",
    "cold_hot",
    "discomfort",
    "hungry",
    "lonely",
    "scared",
    "tired"
]


def get_intensity(score):
    if score < 0.40:
        return "low"
    elif score < 0.70:
        return "medium"
    else:
        return "high"


def predict_audio(audio_path, audio_model):

    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)

    print("\n================ AUDIO DEBUG ================")
    print("Audio path:", audio_path)
    print("Audio length (seconds):", round(len(y) / sr, 2))
    print("Sample rate:", sr)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    print("Original MFCC shape:", mfcc.shape)

    # Ensure exactly 110 frames
    if mfcc.shape[1] < 110:
        pad_width = 110 - mfcc.shape[1]
        mfcc = np.pad(
            mfcc,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )
        print("MFCC padded to:", mfcc.shape)

    else:
        mfcc = mfcc[:, :110]
        print("MFCC cropped to:", mfcc.shape)

    # Normalize
    max_value = np.max(np.abs(mfcc))
    print("MFCC max value before normalization:", max_value)

    if max_value != 0:
        mfcc = mfcc / max_value

    # Reshape for CNN
    mfcc = np.expand_dims(mfcc, axis=-1)   # (40,110,1)
    mfcc = np.expand_dims(mfcc, axis=0)    # (1,40,110,1)

    print("Final model input shape:", mfcc.shape)

    # Prediction
    prediction = audio_model.predict(mfcc, verbose=0)

    print("RAW AUDIO PREDICTIONS:")
    print(prediction)
    print("============================================\n")

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    cry_reason = CRY_CLASSES[predicted_index]
    intensity = get_intensity(confidence)

    return {
        "cry_reason": cry_reason,
        "cry_confidence": round(confidence, 4),
        "cry_intensity": intensity,
        "raw_prediction": prediction.tolist()
    }