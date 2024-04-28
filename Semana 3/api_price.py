from flask import Flask, request, jsonify
import joblib
import numpy as np

# Cargar el modelo entrenado
model = joblib.load('model_rf.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Extraer características desde el JSON
    features = [
        data['Year'],
        data['Mileage'],
        data['State_encoded'],
        data['Make_encoded'],
        data['Model_encoded']
    ]
    
    # Predecir el precio usando el modelo cargado
    price_log = model.predict([features])[0]  # Suponiendo que el modelo espera una matriz 2D
    price = np.exp(price_log)  # Convertir de log-precio a precio

    # Devolver la predicción como un JSON
    return jsonify({
        'predicted_price': price
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)