import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

df = pd.read_csv('data/sentimentos.csv')

modelo = make_pipeline(TfidfVectorizer(), MultinomialNB())

modelo.fit(df['frase'], df['sentimento'])

joblib.dump(modelo, '../modelo_sentimentos.pkl')
print("Modelo treinado com sucesso!")