import tensorflow as tf
import tensorflow.keras as kr

def create_model(word_count, input_length, pre_train_file=None):
    model = kr.models.Sequential()

    model.add(kr.layers.Embedding(input_dim=word_count, output_dim=64, input_length=input_length, name='embed1'))
    model.add(kr.layers.SimpleRNN(units=128, name='rnn1'))
    model.add(kr.layers.Dense(units=word_count, activation='softmax', name='fc1'))

    if pre_train_file is not None:
        model.load_weights(pre_train_file, by_name=True)

    return model