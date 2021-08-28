# text-generation
#import dependencies
import numpy
import sys
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layer import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
#load data
#loading data and opening our input data in the form of a txt file
#Project Gutenberg/berg is where the data can be found(Just Google it!)
file=open("frankenstein-2.txt").read()
#tokenization
#standardization





































