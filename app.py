from flask import Flask, request, jsonify
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
import numpy as np

# Charger les modèles
try:
    chain_classifier = joblib.load('chain_classifier.pkl')
    sbert_model = SentenceTransformer('sbert_model')
  
   # mlb =joblib.load('mlb.pkl')
    print("Les modèles sont chargés correctement.")
except Exception as e:
    print(f"Erreur lors du chargement des modèles : {e}")

app = Flask(__name__)

@app.route('/')
def home():
    return "API de prédiction avec SBERT et Classifier Chain est en cours d'exécution."

@app.route('/predict/<string:question>')
def predict(question):
   
        
        # Encoder les phrases avec SBERT
    SBERT_embeddings = sbert_model.encode([question])
        
        # Faire des prédictions
    predictions = chain_classifier.predict(SBERT_embeddings)
        
      
       # tags = mlb.inverse_transform(predictions)
        
        
    return jsonify({'predictions': predictions})
@app.route('/salut_perso/<string:first_name>')
def salut_toi(first_name):
    return f"Salut {first_name} !"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')