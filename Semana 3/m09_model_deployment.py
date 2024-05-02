import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sys
import os

# Función para cargar datos
def load_and_preprocess_data():
    # Cargar datos
    dataTraining = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTrain_carListings.zip')
    dataTraining['Price'] = np.log(dataTraining['Price'])
    dataTraining['Mileage'] = np.log(dataTraining['Mileage'])

    # Preparar características
    X = dataTraining[['Year', 'Mileage', 'State', 'Make', 'Model']]
    le_State = LabelEncoder()
    le_Make = LabelEncoder()
    le_Model = LabelEncoder()
    X['State_encoded'] = le_State.fit_transform(X['State'])
    X['Make_encoded'] = le_Make.fit_transform(X['Make'])
    X['Model_encoded'] = le_Model.fit_transform(X['Model'])
    X = X[['Year', 'Mileage', 'State_encoded', 'Make_encoded', 'Model_encoded']]
    y = dataTraining['Price']

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=12)

    # Guardar encoders para uso futuro
    joblib.dump(le_State, 'le_State.pkl')
    joblib.dump(le_Make, 'le_Make.pkl')
    joblib.dump(le_Model, 'le_Model.pkl')

    return X_train, X_test, y_train, y_test

# Función para entrenar el modelo
def train_model(X_train, y_train):
    model = RandomForestRegressor(max_features=4, n_estimators=100, max_depth=19, random_state=1, min_samples_split=17, n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, 'model_rf.pkl')  # Guardar el modelo
    return model

# Función para realizar predicciones
def predict_price(features):
    model = joblib.load('model_rf.pkl')
    le_State = joblib.load('le_State.pkl')
    le_Make = joblib.load('le_Make.pkl')
    le_Model = joblib.load('le_Model.pkl')

    # Codificar características entrantes
    features['State'] = le_State.transform([features['State']])
    features['Make'] = le_Make.transform([features['Make']])
    features['Model'] = le_Model.transform([features['Model']])
    features_df = pd.DataFrame([features])

    # Realizar predicción
    price_log = model.predict(features_df)[0]
    price = np.exp(price_log)  # Convertir el precio logarítmico de vuelta a su forma original

    return price

# Si el script es el principal ejecutado, entrenar el modelo
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model = train_model(X_train, y_train)
    print("Model trained and saved successfully.")