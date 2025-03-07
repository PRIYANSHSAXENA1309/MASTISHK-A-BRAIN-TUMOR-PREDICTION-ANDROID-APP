import cv2
import numpy as np
from keras.models import load_model

model = load_model("D:\BrainTumorModel1.keras")

num = "0"
image_path = f"D:/pred{num}.jpg"

image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"Image not found at {image_path}")

image = cv2.resize(image, (240, 240))
image = image.astype('float32') / 255.0  
image = np.expand_dims(image, axis=0)  

prediction = model.predict(image)

predicted_class = 1 if prediction[0] > 0.5 else 0 

class_name = {0: "No Tumor", 1: "Yes Tumor"}

print(f"Prediction: {class_name[predicted_class]}")