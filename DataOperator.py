import pickle

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

def load_text_tokenizer(tokenizer_path):
  with open(tokenizer_path, 'rb') as tokenizer_file:
      return pickle.load(tokenizer_file)

def save_text_tokenizer(tokenizer, tokenizer_path):
  with open(tokenizer_path, 'wb') as tokenizer_file:
      pickle.dump(tokenizer, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)

def load_sequence_len(sequence_len_path):
    sequence_len_file = open(sequence_len_path, 'r')
    line = sequence_len_file.readline()
    sequence_len = int(line)
    sequence_len_file.close()
    return sequence_len

def save_sequence_len(sequence_len, sequence_len_path):
    sequence_len_file = open(sequence_len_path, 'w')
    sequence_len_file.write('{0}'.format(sequence_len))
    sequence_len_file.close()

def load_train_data(train_data_path):
    train_data_file = open(train_data_path, 'r')
    train_data_text = train_data_file.read()
    train_data_file.close()
    train_data_text = train_data_text.lower()

    tokenizer = kr.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts([train_data_text])

    sequence_list = list()
    for line in train_data_text.split('\n'):
        encoded = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(encoded)):
            sequence_list.append(encoded[:i + 1])

    word_count = len(tokenizer.word_index) + 1
    max_sequence_len = max(len(sequence) for sequence in sequence_list)

    sequence_list = kr.preprocessing.sequence.pad_sequences(sequence_list, maxlen=max_sequence_len, padding='pre')
    sequence_list = np.array(sequence_list)

    train_text = sequence_list[:, :-1]
    train_label = kr.utils.to_categorical(sequence_list[:, -1], num_classes=word_count)

    return tokenizer, word_count, max_sequence_len, train_text, train_label