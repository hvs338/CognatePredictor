import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


encoding_table = pd.read_excel("/Users/haroldsmith/Desktop/CognatePredictor/Encoding_Table.xlsx").head(25)
encoding_table = encoding_table.fillna(0).to_dict()
keys = list(encoding_table.keys())[1:]

n = len(encoding_table.values())
values = []
for i in range(1,n):
    #print(np.array(encoding_table.values()))
    okay = []
    stuff = (list(encoding_table.values())[i])
    for j in range(len(stuff)):
        #print(stuff[j])
        okay.append(int((stuff[j])))
    values.append(okay)
    
values = np.array(values)[1:]

encoding_dict = {}
for j in range(len(values)):
    encoding_dict[keys[j]] = values[j]

symbols = list(encoding_dict.keys())
encoding_list = list(encoding_dict.values())
encoding_list = np.asarray(encoding_list)


def encoding(words:list)->list:
    print()
    print("**********************")
    print("[INFO] ENCODING DATA")
    print("**********************")
    print()
    word_translated = []
    for word in words:
        if(type(word) != float):
            chars = list(word)
           # print(chars)
            k = 0
            phenomes_endcode = []
            for char in chars:
                if(char in symbols):
                #print(encoding_dict[char])
                    phenomes_endcode.append(encoding_dict[char])
            word_translated.append(phenomes_endcode)
    
    max_len = 25 # found this by avg_len + 3*std) but I need this the same for training and testing set
    sequences = pad_sequences(word_translated, maxlen = max_len ,padding= 'post')
    #print(sequences.shape)
    return sequences


def meaningful(words:list):
    print("**********************")
    print("RETURNING MEANINGFUL CHARACTERS")
    print("**********************")
    print()
    word_translated = []
    for word in words:
        if(type(word) != float):
            chars = list(word)
            # print(chars)
            k = 0
            phenomes_endcode = []
            for char in chars:
                if(char in symbols):
                #print(encoding_dict[char])
                    phenomes_endcode.append(char)
            word_translated.append(phenomes_endcode)
    return word_translated
#max_len = 25 # found this by avg_len + 3*std) but I need this the same for training and testing set
#sequences = pad_sequences(word_translated, maxlen = max_len ,padding= 'post')
#print(sequences.shape)

