import cv2
import numpy as np


def verify_baby(frame_path, baby_verify_model):
    """
    Baby vs Non-Baby verification using our trained CNN model.
    Baby = 1
    Non-baby = 0
    """

    img = cv2.imread(frame_path)

    if img is None:
        print("Baby verification: frame not found")
        return False

    img = cv2.resize(img, (160, 160))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = baby_verify_model.predict(img, verbose=0)[0][0]

    print("Baby verification score:", prediction)

    if prediction >= 0.5:
        return True

    return False