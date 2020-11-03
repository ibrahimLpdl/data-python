import pandas as pd
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
import streamlit as st
#faire une installation de stop_words
from stop_words import get_stop_words
#permet de sauvegarder les donnees dans un fichier pkl
import joblib 

labelData = pd.read_csv("labels.csv")

labelData.head()

labelData = labelData.drop("Unnamed: 0",1)

clf = make_pipeline(
  TfidfVectorizer(stop_words=get_stop_words('en')),
  OneVsRestClassifier(SVC(kernel='linear', probability=True))
)


joblib.dump(clf, 'resultCLF.pkl')