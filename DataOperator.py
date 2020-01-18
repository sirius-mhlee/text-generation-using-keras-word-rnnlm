import pickle

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

import Configuration as cfg

def load_text_tokenizer(tokenizer_path):
  with open(tokenizer_path, 'rb') as tokenizer_file:
      return pickle.load(tokenizer_file)

def save_text_tokenizer(tokenizer, tokenizer_path):
  with open(tokenizer_path, 'wb') as tokenizer_file:
      pickle.dump(tokenizer, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)

def load_train_data(train_data_path):
    train_data_file = open(train_data_path, 'r')
    train_data_text = train_data_file.read()
    train_data_file.close()
    train_data_text = train_data_text.lower().replace('\r', '').replace('\n', ' ')

    tokenizer = kr.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts([train_data_text])
    encoded_train_data_text = tokenizer.texts_to_sequences([train_data_text])[0]

    sequence_list = list()
    for pos in range(cfg.max_sequence_len + 1, len(encoded_train_data_text)):
        encoded = encoded_train_data_text[pos - (cfg.max_sequence_len + 1):pos]
        for i in range(1, len(encoded)):
            sequence_list.append(encoded[:i + 1])

    word_count = len(tokenizer.word_index) + 1

    sequence_list = kr.preprocessing.sequence.pad_sequences(sequence_list, maxlen=cfg.max_sequence_len + 1, padding='pre', truncating='pre')
    sequence_list = np.array(sequence_list)

    train_text = sequence_list[:, :-1]
    train_label = kr.utils.to_categorical(sequence_list[:, -1], num_classes=word_count)

    return tokenizer, word_count, train_text, train_label