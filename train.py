import pandas as pd
import numpy as np
from rich.console import Console
from tqdm import tqdm
import re
from nltk import pos_tag
from joblib import dump, load
import nltk
from nltk.corpus import stopwords
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

# seed_everything(42)

tqdm.pandas()


def lemmatize_words(text, lemmatizer, wordnet_map):
    pos_tagged_text = pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])





def remove_punctuation(text):
    """custom function to remove the punctuation"""
    PUNCT_TO_REMOVE = string.punctuation
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))



def eval_and_print_metrics(clf, X_train, y_train, X_test, y_test):
    print("Number of training samples:", len(X_train))
    print("Unlabeled samples in training set:",
          sum(1 for x in y_train if x == -1))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    Console().log(f"----------------------------------------------------")
    Console().print(f"y_pred =====> {y_pred}")
    Console().print(f"y_pred =====> {y_test}")
    Console().log(f"----------------------------------------------------")
    print("Micro-averaged F1 score on test set: "
          "%0.3f" % f1_score(y_test, y_pred, average='micro'))
    print("-" * 10)
    print()
    dump(clf, 'model.joblib')




def main():
    nltk.download('wordnet')
    nltk.download('stopwords')
    # some vars. for text processing
    # lemmatizer = WordNetLemmatizer()
    # wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    #
    # # read csv
    # data = pd.read_csv('NLP.csv')
    #
    # # text pre-processing
    # data["cmd"] = data["NLP Statements"].progress_apply(lambda x: x.strip().lower())
    # data["cmd"] = data["cmd"].progress_apply(lambda text: remove_punctuation(text))
    # data["cleaned"] = data["cmd"].progress_apply(lambda x: " ".join([k for k in x.split() if k not in stopwords.words('english')]))
    # data["lemmatized"] = data["cleaned"].progress_apply(lambda x: lemmatize_words(x, lemmatizer, wordnet_map))
    #
    # # save the states of lemmatizer and dictionaries
    # dump(lemmatizer, "lemmatizer.pkl")
    # dump(wordnet_map, "wordnet_map.pkl")
    #
    # Console().print(data["lemmatized"])
    #
    # data.to_csv('NLP_cleaned.csv', index=False)
    #
    # df = pd.read_csv('NLP_cleaned.csv', skipinitialspace=True, skip_blank_lines=True)
    # df.drop([k for k in df.columns if k not in ["Commands", "lemmatized"]], axis=1, inplace=True)
    # Console().print(df.columns)
    # le = LabelEncoder()
    # # df["Commands"] = le.fit_transform(df["Commands"])
    # df["encoded_labels"] = le.fit_transform(df["Commands"])
    # # save the label encoder
    # dump(le, "label_encoder.pkl")
    # # save encoded labels to csv
    # df.to_csv('NLP_cleaned.csv', index=False)

    # =====================================================================================================

    df = pd.read_csv("NLP_cleaned.csv", skipinitialspace=True, skip_blank_lines=True)
    df["Commands"] = df["encoded_labels"]

    # Console().print(f"UNIQUE labels --> {len(le.classes_)}", style="cyan on black")
    Console().log(f"{df.columns.tolist()}")
    # # split the dataset
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)
    Console().rule(title=f'[color(154)]train [color(215)]{train_df.shape}', style='magenta on black', characters='=')
    Console().rule(title=f'[color(154)]test [color(215)]{test_df.shape}', style='magenta on black', characters='=')

    Console().log(f"xtrain cols. {train_df.columns.tolist()}")
    xtrain, ytrain = train_df["lemmatized"].values, train_df["Commands"].values
    xtest, ytest = test_df["lemmatized"].values, test_df["Commands"].values

    Console().log(f"================================================================")
    Console().log(f"xtrain shape --->   {xtrain.shape}")
    Console().log(f"xtest shape --->   {xtest.shape}")
    Console().log(f"ytrain shape ---> {ytrain.shape}")
    Console().log(f"ytest shape ---> {ytest.shape}")
    Console().log(f"YTRAIN unique tokens ---> {np.unique(ytrain)}")
    Console().log(f"YTEST unique tokens ---> {np.unique(ytest)}")
    Console().log(f"================================================================")

    # train
    # Parameters
    sdg_params = dict(alpha=1e-5, penalty='l2', loss='log')
    vectorizer_params = dict(ngram_range=(1, 5), min_df=5, max_df=0.8)

    pipeline = Pipeline([

                            ('vect', CountVectorizer(**vectorizer_params)),
                            ('tfidf', TfidfTransformer()),
                            ('clf', SGDClassifier(**sdg_params)),
                        ])



    print("Supervised SGDClassifier on 100% of the data:")
    eval_and_print_metrics(pipeline, xtrain, ytrain, xtest, ytest)




if __name__=='__main__':
    main()