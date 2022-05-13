from pathlib import Path

import pandas as pd
import numpy as np

from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


def round_val(val):
    return round(float(val), 4)


def replace_empty(values):
    result = list()
    for row in range(values.shape[0]):
        if np.isnan(values[row]):
            result = list()
        else:
            result.append(values[row])
    return array(result)


def split_dataset(data):
    return data[0], replace_empty(data[1:]).astype('float32')


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
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


def prepare_reconciliation(data, split_index, n_steps):
    for index in range(len(data) - split_index, len(data)):
        value = data[index - n_steps: index].astype('float32')
        yield data[index], array(value)


def predict_and_test(model, data, split_index, n_steps, n_features):
    correct, predict = list(), list()
    for key, fit_data in prepare_reconciliation(data, split_index, n_steps):
        correct.append(key)
        x_input = fit_data.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)[0][0]
        predict.append(round_val(yhat))
    return correct, predict


def prepare_future_data(data, n_steps):
    return array(data[len(data) - n_steps:].astype('float32'))


def predict_future(model, data, split_index, n_steps, n_features):
    result = list()
    for _ in range(split_index):
        fit_data = prepare_future_data(data, n_steps)
        x_input = fit_data.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)[0][0]
        result.append(round_val(yhat))
        data = np.concatenate([data, array([yhat])])
    return result


def add_empty_cols(df, cols_volume):
    df = df.T
    last_year = df.columns.values[-1]
    for i in range(1, cols_volume + 1):
        df[str(last_year + i)] = pd.Series(dtype=float)
    return df.T


def get_new_col_index(current):
    return current * 2 + 1


def add_cols(df, new_data, new_cols_vol):
    empty_cells = df.shape[0] - new_cols_vol - 1
    df = add_empty_cols(df, new_cols_vol)
    for col_index in df.columns:
        title = df[col_index].values[0]
        col_data = [f"{title} прогноз", *[""] * empty_cells, *[str(round_val(v)).replace('.', ',') for v in new_data[title]]]
        new_index = get_new_col_index(col_index)
        df.insert(new_index, f"{col_index}*", col_data)
    return df


def write_to_csv(df):
    filepath = Path('../data/out.csv')
    df.T.to_csv(filepath, encoding='windows-1251', index=False, sep=";")


def run():
    xl = pd.ExcelFile('../data/Статистические_данные_показателей_СЭР.xlsx').parse("Прогнозируемые показатели")
    df = xl.T
    reconciliation_index = 4
    n_steps = 9
    n_features = 1
    result = dict()
    for col_index in df.columns:
        col = df[col_index]
        title, train = split_dataset(col.values)
        X, y = split_sequence(train, n_steps)
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        model = get_model(n_steps, n_features, X, y)
        correct, predict = predict_and_test(model, train, reconciliation_index, n_steps, n_features)
        future = predict_future(model, train, reconciliation_index, n_steps, n_features)
        result[title] = [*predict, *future]
        print(f"{title} predicted")
    df = add_cols(df, result, reconciliation_index)
    write_to_csv(df)


if __name__ == '__main__':
    run()
