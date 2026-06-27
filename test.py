from utils.mapping import CLASS_TO_DISEASE
from utils.mapping import DISEASE_NAMES
from utils.preprocess import preprocess_image

print(CLASS_TO_DISEASE)
print(DISEASE_NAMES['nv'])

image = 'sample_images/ISIC_0024306.jpg'
image = preprocess_image(image)

print(image.shape)
print(image.dtype)