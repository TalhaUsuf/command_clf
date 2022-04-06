import pandas as pd
import numpy as np
from rich.console import Console
from tqdm import tqdm
import re
from nltk import pos_tag
from joblib import dump, load
import nltk
from nltk.corpus import stopwords
from fastapi import FastAPI
from typing import List, Union
from pydantic import BaseModel
from sklearn.metrics import f1_score
import string
# from pytorch_lightning import seed_everything
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()




class RESP_MODEL(BaseModel):
    cmd_input : str
    match : str
    reference_trained : List[str]
    idx_match : int


tqdm.pandas()



def lemmatize_words(text, lemmatizer, wordnet_map):
    pos_tagged_text = pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])





def remove_punctuation(text):
    """custom function to remove the punctuation"""
    PUNCT_TO_REMOVE = string.punctuation
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


app = FastAPI()


origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/{sent}", response_model=RESP_MODEL)
async def pred(sent : str):
    # nltk.download('wordnet')
    # nltk.download('stopwords')

    # some vars. for text processing
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}


    # loading the saved states
    pipeline = load("model.joblib")
    le = load("label_encoder.pkl")
    lemmatizer = load("lemmatizer.pkl")
    wordnet_map = load("wordnet_map.pkl")

    # turn input sentence to dataframe
    Console().log(f"received --> {sent}")
    data = {"NLP Statements" : [sent],
          "Commands" : ["None"]
    }
    data = pd.DataFrame.from_dict(data)
    Console().log(f"dataframe ====> \n{data}")

    # # text pre-processing
    data["cmd"] = data["NLP Statements"].progress_apply(lambda x: x.strip().lower())
    data["cmd"] = data["cmd"].progress_apply(lambda text: remove_punctuation(text))
    data["cleaned"] = data["cmd"].progress_apply(lambda x: " ".join([k for k in x.split() if k not in stopwords.words('english')]))
    data["lemmatized"] = data["cleaned"].progress_apply(lambda x: lemmatize_words(x, lemmatizer, wordnet_map))

    Console().log(f"after cleaning ---> \n {data}")

    Console().log(f"model loaded \n\n {pipeline}")

    prediction = pipeline.predict(data["lemmatized"])
    Console().log(f"prediction index [red] matched ---> [green]{prediction}")
    Console().log(f"target sentences to match with \n\n {le.classes_}")
    Console().log(f"Matching command ===> [yellow] {le.classes_[prediction.squeeze()]}")

    return {
        "cmd_input" : sent,
        "match" : le.classes_[prediction.squeeze()],
        "reference_trained" : le.classes_.tolist(),
        "idx_match" : prediction.squeeze()
    }