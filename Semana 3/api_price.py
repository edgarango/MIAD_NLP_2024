from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from flask_cors import CORS
from m09_model_deployment import predict_price

# Inicia flask
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

# API
api = Api(
    app, 
    version='1.0', 
    title='Prediccion Precios Vehiculos',
    description='API usa un modelo Random Forest para pedecir precios de vehiculos'
)

# Create a namespace
ns = api.namespace('predict', description='Predice precios vehiculos')

# Define the parser for incoming request arguments
parser = api.parser()
parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='Modelo (año) vehiculo', 
    location='args')
parser.add_argument(
    'Mileage', 
    type=float, 
    required=True, 
    help='Cantidad millas rodadas', 
    location='args')
parser.add_argument(
    'State', 
    type=str, 
    required=True, 
    help='Estado donde esta registrado', 
    location='args')
parser.add_argument(
    'Make', 
    type=str, 
    required=True, 
    help='Marca vehiculo', 
    location='args')
parser.add_argument(
    'Model', 
    type=str, 
    required=True, 
    help='modelo vehiculo', 
    location='args')

# Define resource fields
resource_fields = api.model('Resource', {
    'predicted_price': fields.Float,
})

# Create a class for the prediction resource
@ns.route('/')
class VehiclePriceApi(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        # Extract query parameters
        args = parser.parse_args()
        # Use the imported function to predict the price
        predicted_price = predict_price(**args)
        # Return the predicted price
        return {'predicted_price': predicted_price}

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)