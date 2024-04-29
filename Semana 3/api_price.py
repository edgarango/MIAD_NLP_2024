from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Carga del modelo y encoders
model = joblib.load('model_rf.pkl')
le_State = joblib.load('le_State.pkl')
le_Make = joblib.load('le_Make.pkl')
le_Model = joblib.load('le_Model.pkl')

def preprocess_data(df):
    # Aquí deberías incluir la lógica de preprocesamiento aplicada antes del entrenamiento
    df['Mileage'] = np.log(df['Mileage'])
    return df

def encode_features(df):
    df['State_encoded'] = le_State.transform(df['State'])
    df['Make_encoded'] = le_Make.transform(df['Make'])
    df['Model_encoded'] = le_Model.transform(df['Model'])
    return df[['Year', 'Mileage', 'State_encoded', 'Make_encoded', 'Model_encoded']]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data, index=[0])

    # Preprocesamiento y codificación de características
    df_processed = preprocess_data(df)
    df_encoded = encode_features(df_processed)
    
    # Predicción
    prediction = model.predict(df_encoded)
    
    # Devolución de la predicción
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)