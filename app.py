from flask import Flask, request, jsonify
import pickle  # Remplace joblib par pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
import numpy as np

# Charger les modèles
try:
    # Charger le modèle ClassifierChain avec pickle
    with open('chain_classifier.pkl', 'rb') as file:
        chain_classifier = pickle.load(file)
    
    # Charger le modèle SBERT avec pickle
    with open('sbert_model.pkl', 'rb') as file:
        sbert_model = pickle.load(file)

    # Charger le MultiLabelBinarizer avec pickle (si nécessaire)
    # with open('mlb.pkl', 'rb') as file:
    #     mlb = pickle.load(file)
    
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
    
    # Convertir les prédictions en tags (si nécessaire)
    # tags = mlb.inverse_transform(predictions)
    
    return jsonify({'predictions': predictions.tolist()})  # Utiliser .tolist() pour rendre JSON sérialisable

@app.route('/salut_perso/<string:first_name>')
def salut_toi(first_name):
    return f"Salut {first_name} !"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
