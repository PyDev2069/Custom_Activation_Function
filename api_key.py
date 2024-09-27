from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from iris_neural import custom_activation, output_activation  
app = Flask(__name__)
model = tf.keras.models.load_model('api.h5', custom_objects={
    'custom_activation': custom_activation,
    'output_activation': output_activation
})
@app.route('/predict', methods=['POST'])
def predict():
    try:        
        data = request.get_json()        
        input_data = np.array(data['input']).reshape(1, -1)        
        prediction = model.predict(input_data)        
        predicted_class = np.argmax(prediction, axis=1)       
        return jsonify({
            'prediction': int(predicted_class[0]),
            'probabilities': prediction.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True)
