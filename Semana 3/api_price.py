from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from flask_cors import CORS
#from models.m09_model_deployment import predict_price
from m09_model_deployment import predict_price  # Import the prediction function

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

# Initialize API with Flask app
api = Api(
    app, 
    version='1.0', 
    title='Vehicle Price Prediction API',
    description='API uses a Random Forest model to predict used vehicle prices'
)

# Create a namespace
ns = api.namespace('predict', description='Predict vehicle prices')

# Define the parser for incoming request arguments
parser = api.parser()
parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='Year of the vehicle', 
    location='args')
parser.add_argument(
    'Mileage', 
    type=float, 
    required=True, 
    help='Mileage of the vehicle', 
    location='args')
parser.add_argument(
    'State_encoded', 
    type=int, 
    required=True, 
    help='Encoded state of vehicle registration', 
    location='args')
parser.add_argument(
    'Make_encoded', 
    type=int, 
    required=True, 
    help='Encoded make of the vehicle', 
    location='args')
parser.add_argument(
    'Model_encoded', 
    type=int, 
    required=True, 
    help='Encoded model of the vehicle', 
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