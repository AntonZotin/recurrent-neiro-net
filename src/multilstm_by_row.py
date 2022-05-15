from pathlib import Path

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
    return replace_empty_cols(data[1:]).astype('float32')


def split_sequences(sequences, n_steps, predict_index):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences) - 1:
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, predict_index]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def get_model(n_steps, n_features, train_x, train_y):
    model = Sequential()
    model.add(LSTM(150, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y, epochs=3000, verbose=0)
    return model


def prepare_reconciliation(data, split_index, n_steps, predict_index):
    for index in range(len(data) - split_index, len(data)):
        check_data = data[index][predict_index].astype('float32')
        fit_data = data[index - n_steps: index].astype('float32')
        yield check_data, fit_data


def predict_exists(model, data, split_index, n_steps, n_features, predict_index):
    correct, predict = list(), list()
    for check_data, fit_data in prepare_reconciliation(data, split_index, n_steps, predict_index):
        correct.append(check_data)
        x_input = fit_data.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)[0][0]
        predict.append(round_val(yhat))
    return correct, predict


def prepare_future_data(data, n_steps):
    return array(data[len(data) - n_steps:].astype('float32'))


def predict_future(model, data, n_steps, n_features):
    fit_data = prepare_future_data(data, n_steps)
    x_input = fit_data.reshape((1, n_steps, n_features))
    return model.predict(x_input, verbose=0)[0][0]


def get_new_col_index(current):
    return current * 2 + 1


def add_cols(df, new_data, new_cols_vol):
    empty_cells = df.shape[0] - new_cols_vol - 1
    for col_index in df.columns:
        title = df[col_index].values[0]
        col_data = [f"{title} прогноз", *[""] * empty_cells, *[str(round_val(v)).replace('.', ',') for v in new_data[title]]]
        new_index = get_new_col_index(col_index)
        df.insert(new_index, f"{col_index}*", col_data)
    return df


def read_from_xls():
    return pd.ExcelFile('../data/Статистические_данные_показателей_СЭР.xlsx').parse("Прогнозируемые показатели")[:3].T


def write_to_csv(df):
    filepath = Path('../data/out2.csv')
    df.T.to_csv(filepath, encoding='windows-1251', index=False, sep=";")


def add_value_or_create_section(dictionary, section, key, value):
    if section in dictionary:
        if key in dictionary[section]:
            dictionary[section][key] += value
        else:
            dictionary[section][key] = value
    else:
        dictionary[section] = {key: value}


def run():
    try:
        reconciliation_index = 4
        n_steps = 9
        df = read_from_xls()
        train = split_dataset(df.values)
        result = {}
        for col_index in df.columns:
            title = df[col_index].values[0]
            X, y = split_sequences(train, n_steps, col_index)
            n_features = X.shape[2]
            model = get_model(n_steps, n_features, X, y)
            correct, predicted = predict_exists(model, train, reconciliation_index, n_steps, n_features, col_index)
            # add_value_or_create_section(result, title, 'predict', predicted)
            # add_value_or_create_section(result, title, 'model', model)
            print(f"{title}: Correct: {correct}, Predict: {predicted}")
            result[title] = predicted
        # df = add_cols(df, result, reconciliation_index)
        # write_to_csv(df)

        # for r in range(reconciliation_index):
        #     for col_index in df.columns:
        #         title = df[col_index].values[0]
        #         X, y = split_sequences(train, n_steps, col_index)
        #         n_features = X.shape[2]
        #         model = result[title]['model']
        #         future = predict_future(model, train, n_steps, n_features)
        #         add_value_or_create_section(result, title, 'future', future)
    except Exception as e:
        ax = 1
        raise e


if __name__ == '__main__':
    run()
