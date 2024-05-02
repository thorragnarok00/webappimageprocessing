from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load your trained model
model_path = 'C:/xampp/htdocs/Mobile/model/model_name.keras'
model = tf.keras.models.load_model(model_path)

# Define your class labels
class_labels = ['Apple', 'Orange']  

# Function to preprocess image before feeding it to the model
def preprocess_image(image):
    image = image.resize((150, 150)) 
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']
    image = Image.open(image_file)
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(np.expand_dims(processed_image, axis=0))

    # Debug logging
    print('Prediction scores:', prediction)

    # Get the predicted class label
    predicted_class_index = int(np.round(prediction)[0][0])
    predicted_class = class_labels[predicted_class_index]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 5000, debug=True)  # Run the Flask app