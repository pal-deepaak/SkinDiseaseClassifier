import numpy as np

from tensorflow.keras.models import load_model

from utils.preprocess import preprocess_image
from utils.mapping import DISEASE_DESCRIPTIONS
from utils.mapping import DISEASE_NAMES
from utils.mapping import CLASS_TO_DISEASE

MODEL_PATH = 'models/resnet50_finetuned_model.keras'
model = load_model(MODEL_PATH)

def predict(image_path: str)-> dict: 
    image = preprocess_image(image_path)
    predictions = model.predict(image, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    confidence = float(predictions[predicted_class])
    disease_code = CLASS_TO_DISEASE[predicted_class]
    top3_indices = np.argsort(predictions)[::-1][:3]
    top3_predictions = []
    for idx in top3_indices:
        code = CLASS_TO_DISEASE[idx]
        top3_predictions.append({
            "disease_code" : code,
            "disease_name" : DISEASE_NAMES[code],
            "confidence" : float(predictions[idx])
        })

    result = {
        "disease_code" : disease_code,
        "disease_name" : DISEASE_NAMES[disease_code],
        "description" : DISEASE_DESCRIPTIONS[disease_code],
        "confidence" : confidence,
        "probabilities" : predictions,
        "top3" : top3_predictions
    }
    return result

if __name__ == "__main__":

    image_path = "sample_images/ISIC_0024306.jpg"

    result = predict(image_path)

    print(result)