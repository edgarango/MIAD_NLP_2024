from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
api = Api(
    app, 
    version='1.0', 
    title='Prediccion Genero Peliculas',
    description='API Predice genero de peliculas con el plot(resumen pelicula)'
)

# Enable CORS for all routes and origins
from flask_cors import CORS
CORS(app)

# Carga del modelo
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Titulos
ns = api.namespace('Prediccion', description='Genero de Peliculas')

# Definir los argumentos del parser
parser = api.parser()
parser.add_argument(
    'plot', 
    type=str, 
    required=True, 
    help='Resumen de la pelicula', 
    location='args'
)

# Definir los campos del recurso
resource_fields = api.model('Resource', {
    'p_Action': fields.Float,
    'p_Adventure': fields.Float,
    'p_Animation': fields.Float,
    'p_Biography': fields.Float,
    'p_Comedy': fields.Float,
    'p_Crime': fields.Float,
    'p_Documentary': fields.Float,
    'p_Drama': fields.Float,
    'p_Family': fields.Float,
    'p_Fantasy': fields.Float,
    'p_Film-Noir': fields.Float,
    'p_History': fields.Float,
    'p_Horror': fields.Float,
    'p_Music': fields.Float,
    'p_Musical': fields.Float,
    'p_Mystery': fields.Float,
    'p_News': fields.Float,
    'p_Romance': fields.Float,
    'p_Sci-Fi': fields.Float,
    'p_Short': fields.Float,
    'p_Sport': fields.Float,
    'p_Thriller': fields.Float,
    'p_War': fields.Float,
    'p_Western': fields.Float
})

# Crear la clase para el recurso de predicción
@ns.route('/')
class GenrePredictionApi(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        # Extraer los parámetros de la consulta
        args = parser.parse_args()
        plot = args['plot']
        
        # Vectorizar la sinopsis
        X_dtm = vectorizer.transform([plot])
        
        # Realizar la predicción
        prediction = model.predict_proba(X_dtm).toarray()[0]
        
        # Crear el resultado como un diccionario
        cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
                'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
                'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']
        result = dict(zip(cols, prediction))
        
        return result

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)