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
    title='Predicción de Género de Películas',
    description='API predice género de películas con el título y el plot (resumen película)'
)

# Enable CORS for all routes and origins
from flask_cors import CORS
CORS(app)

# Carga del modelo
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Titulos
ns = api.namespace('Prediccion', description='Género de Películas')

# Definir los argumentos del parser
parser = api.parser()
parser.add_argument(
    'title', 
    type=str, 
    required=True, 
    help='Título de la película', 
    location='args'
)
parser.add_argument(
    'plot', 
    type=str, 
    required=True, 
    help='Resumen de la película', 
    location='args'
)

# Definir los campos del recurso
resource_fields = api.model('Resource', {
    'Action': fields.Float,
    'Adventure': fields.Float,
    'Animation': fields.Float,
    'Biography': fields.Float,
    'Comedy': fields.Float,
    'Crime': fields.Float,
    'Documentary': fields.Float,
    'Drama': fields.Float,
    'Family': fields.Float,
    'Fantasy': fields.Float,
    'Film-Noir': fields.Float,
    'History': fields.Float,
    'Horror': fields.Float,
    'Music': fields.Float,
    'Musical': fields.Float,
    'Mystery': fields.Float,
    'News': fields.Float,
    'Romance': fields.Float,
    'Sci-Fi': fields.Float,
    'Short': fields.Float,
    'Sport': fields.Float,
    'Thriller': fields.Float,
    'War': fields.Float,
    'Western': fields.Float
})

# Crear la clase para el recurso de predicción
@ns.route('/')
class GenrePredictionApi(Resource):
    @api.doc(parser=parser)
    def get(self):
        # Extraer los parámetros de la consulta
        args = parser.parse_args()
        plot = args['plot']
        title = args['title']  # Aunque no lo usamos, se agrega para mantener la estructura
        
        # Vectorizar la sinopsis
        X_dtm = vectorizer.transform([plot])
        
        # Realizar la predicción
        prediction = model.predict_proba(X_dtm).toarray()[0]
        
        # Crear el resultado como un diccionario
        cols = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
                'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
        result = dict(zip(cols, prediction))
        
        # Ordenar el resultado de mayor a menor y limitar a 3 decimales
        sorted_result = {k: round(v, 3) for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}
        
        # Seleccionar el top 5
        top_5 = {k: sorted_result[k] for k in list(sorted_result)[:5]}
        
        # Combinar top 5 con todos los géneros
        combined_result = {'Probabilidad Generos': top_5}#, 'all_genres': sorted_result}
        
        return jsonify(combined_result)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)