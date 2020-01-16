import sys

import tensorflow as tf
import tensorflow.keras as kr

import DataOperator as do

import RNNLMNetwork as rn

def generate_sentence(tokenizer, index_to_word, model, current_word, max_sequence_len, generate_num):
    current_word = current_word.lower()
    sentence = current_word

    for _ in range(generate_num):
        encoded = tokenizer.texts_to_sequences([current_word])[0]
        encoded = kr.preprocessing.sequence.pad_sequences([encoded], maxlen=max_sequence_len - 1, padding='pre')

        predict_label_index = model.predict_classes(encoded)
        word = index_to_word[predict_label_index[0]]

        current_word = current_word + ' '  + word
        sentence = sentence + ' ' + word

    return sentence

def main():
    input_model_path = sys.argv[1]
    input_tokenizer_path = sys.argv[2]
    input_sequence_len_path = sys.argv[3]
    input_word = sys.argv[4]
    input_predict_count = int(sys.argv[5])
    
    tokenizer = do.load_text_tokenizer(input_tokenizer_path)

    index_to_word = {}
    for word, index in tokenizer.word_index.items():
        index_to_word[index] = word

    max_sequence_len = do.load_sequence_len(input_sequence_len_path)
    word_count = len(tokenizer.word_index) + 1

    rnnlm_model = rn.create_model(word_count, max_sequence_len - 1, input_model_path)

    print()
    print(generate_sentence(tokenizer, index_to_word, rnnlm_model, input_word, max_sequence_len, input_predict_count))

if __name__ == '__main__':
    main()