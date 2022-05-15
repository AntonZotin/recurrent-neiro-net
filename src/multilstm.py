import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from numpy import array


def round_val(val):
    return round(float(val), 4)


def replace_empty_cols(sequences):
    begin_ix = 0
    for ix in range(sequences.shape[0]):
        if np.isnan(np.sum(sequences[ix])) and ix + 1 <= sequences.shape[0]:
            begin_ix = ix + 1
    return sequences[begin_ix:]


def split_dataset(data):
    return data[0], replace_empty_cols(data[1:]).astype('float32')


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences) - 1:
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def get_model(n_steps, n_features, train_x, train_y):
    model = Sequential()
    model.add(LSTM(150, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(150, activation='relu'))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y, epochs=3000, verbose=0)
    return model


def prepare_reconciliation(data, split_index, n_steps):
    for index in range(len(data) - split_index, len(data)):
        check_data = data[index].astype('float32')
        fit_data = data[index - n_steps: index].astype('float32')
        yield check_data, fit_data


def predict_and_test(model, data, split_index, n_steps, n_features):
    correct, predict = list(), list()
    for check_data, fit_data in prepare_reconciliation(data, split_index, n_steps):
        correct.append(check_data)
        x_input = fit_data.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        predict.append(yhat[0])
    return correct, predict


def prepare_future_data(data, n_steps):
    return array(data[len(data) - n_steps:].astype('float32'))


def predict_future(model, data, split_index, n_steps, n_features):
    result = list()
    for _ in range(split_index):
        fit_data = prepare_future_data(data, n_steps)
        x_input = fit_data.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        result.append(yhat[0])
        data = np.concatenate([data, yhat])
    return result


def read_from_xls():
    return pd.ExcelFile('../data/Статистические_данные_показателей_СЭР.xlsx').parse("Прогнозируемые показатели").T


def run():
    try:
        reconciliation_index = 4
        n_steps = 9
        df = read_from_xls()
        title, train = split_dataset(df.values)
        X, y = split_sequences(train, n_steps)
        n_features = X.shape[2]
        model = get_model(n_steps, n_features, X, y)
        correct, predict = predict_and_test(model, train, reconciliation_index, n_steps, n_features)
        future = predict_future(model, train, reconciliation_index, n_steps, n_features)
        for fi in range(n_features):
            correct_arr, predict_arr, future_arr = list(), list(), list()
            for ri in range(reconciliation_index):
                correct_arr.append(correct[ri][fi])
                predict_arr.append(predict[ri][fi])
                future_arr.append(future[ri][fi])
            print(f"{title[fi]}: Correct: {correct_arr}, Predict: {predict_arr}, Future: {future_arr}")
    except Exception as e:
        ax = 1
        raise e


if __name__ == '__main__':
    run()
