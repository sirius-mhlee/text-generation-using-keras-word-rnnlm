import sys

import tensorflow as tf
import tensorflow.keras as kr

import Configuration as cfg

import DataOperator as do

import RNNLMNetwork as rn

def generate_sentence(tokenizer, index_to_word, model, input_word, generate_num):
    sentence = input_word.lower()

    for _ in range(generate_num):
        encoded = tokenizer.texts_to_sequences([sentence])[0]
        encoded = kr.preprocessing.sequence.pad_sequences([encoded], maxlen=cfg.max_sequence_len, padding='pre', truncating='pre')

        predict_label_index = model.predict_classes(encoded)
        word = index_to_word[predict_label_index[0]]
        sentence = sentence + ' ' + word

    return sentence

def main():
    input_model_path = sys.argv[1]
    input_tokenizer_path = sys.argv[2]
    input_word = sys.argv[3]
    input_predict_count = int(sys.argv[4])
    
    tokenizer = do.load_text_tokenizer(input_tokenizer_path)

    index_to_word = {}
    for word, index in tokenizer.word_index.items():
        index_to_word[index] = word

    word_count = len(tokenizer.word_index) + 1
    rnnlm_model = rn.create_model(word_count, input_model_path)

    print()
    print(generate_sentence(tokenizer, index_to_word, rnnlm_model, input_word, input_predict_count))

if __name__ == '__main__':
    main()