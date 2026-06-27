import cv2
import numpy as np

from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = 224

def preprocess_image(image_path : str) -> np.ndarray:
    
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image Not Found : {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
    image = image.astype(np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image