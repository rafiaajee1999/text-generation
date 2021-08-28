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
#What is tokenization?Tokenization is the process of breaking a stream of text up into words phrases symbols or other meaningful element
def tokenize_words(input):
     input = input.lower
     tokenizer = RegexpTokenizer(r'\wt')
     tokens = tokenizer.tokenize(input)
     filtered = filter(lambda token:token not in stopwords.words('english'),tokens)
     return"".join(filtered)
processed_inputs = tokenize_words(file)
#chars to numbers
chars = sorted(list(set(processed_inputs))
chars_to_num = dict((c, i) for i, c in enumerate(chars))
#check if words to chars or chars to num(?!) has worked?
input_len = len(processed_inputs)
vocab_len = len(chars)
print("Total number of characters:",input_len)
print("Total vocab:",vocab_len)
#seq length
seq_length = 100
x_data = []
y_data = []
#loop through the sequence
for i in range(0, input_len - seq_length, 1):
     in_seq = processed_inputs[i:i+ seq_length]
     out_seq = processed_inputs[i+seq_length]
     x_data.append(char_to_num[char] for char in in_seq)
     y_data.append(char_to_num[out_seq])
n_pattern = len(x_data)
print("Total Patterns:",n_pattern)
#convert input sequence to no array and so on
X = bumpy.reshape(x_data, (n_pattern, seq_length, 1))
X = X/float(vocab_len)
#one_hot encoding
y = np_utils.to_categorical(y_data)
#creating the model
model = sequential()
model.add([LSTM(256, input_shape=(X.shape[1], X.shape[2]),return_sequences= True))
model.add(Dropout(0.2))
model.add(LSTM(256,return_sequence=True))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
#compile the model
model.compile(loss = 'categorical_crossentropy', optimizes ='adam')
#saving weights
filepath = "model_weights_saved.hf5"
Checkpoint = ModelCheckpoint(filepath, monitor ='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks=[checkpoint]
#fit model and lat it train
model.fit(X, y, epochs=4, batch_size=256, callbacks=desired_callbacks)
#recompile model with the saved weights
filename ="model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy',optimizer='adam')
#output of the model back into characters
num_to_char = dict((I, c) for i, c in enumerate(chars))
#random seed to help generate
start =bumpy.random.randint(0, len(x_data)_1)
pattern =x_data(start)
print("Random Seed:")
print("\"",''.join((num_to_char[value] for value in pattern]),"\"")
#generate the text
for i in range(1000):
    x = numpy.reshape(pattern,(1,len(pattern),1))
    x = x\float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern = pattern[1:len(pattern)])
  



































