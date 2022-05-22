from pathlib import Path

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from numpy import array
from numpy.ma import hstack


def round_val(val):
    return round(float(val), 4)


def replace_empty_cols(sequences):
    begin_ix = 0
    for ix in range(sequences.shape[0]):
        if np.isnan(np.sum(sequences[ix])) and ix + 1 <= sequences.shape[0]:
            begin_ix = ix + 1
    return sequences[begin_ix:]


def split_dataset(data):
    title, dataset = data[0], data[1:].astype('float32')
    for col in range(dataset.shape[0]):
        for row in range(dataset.shape[1]):
            if np.isnan(dataset[col][row]):
                return title, dataset, col
    return title, dataset, len(dataset)


def split_sequences(sequences, n_steps, end_prediction_rows, empty_data_index):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences) - 1 or end_ix >= empty_data_index:
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :end_prediction_rows]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def get_model(n_steps, n_input, n_output, train_x, train_y):
    model = Sequential()
    model.add(LSTM(150, activation='relu', return_sequences=True, input_shape=(n_steps, n_input)))
    model.add(LSTM(150, activation='relu'))
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y, epochs=3000, verbose=0)
    return model


def prepare_reconciliation_data(data, split_index, n_steps, empty_data_index):
    for index in range(empty_data_index - split_index, empty_data_index):
        fit_data = data[index - n_steps: index].astype('float32')
        yield fit_data


def predict_exists(model, data, split_index, n_steps, n_input, empty_data_index):
    predict = list()
    for fit_data in prepare_reconciliation_data(data, split_index, n_steps, empty_data_index):
        x_input = fit_data.reshape((1, n_steps, n_input))
        yhat = model.predict(x_input, verbose=0)
        predict.append(yhat[0])
    return predict


def prepare_future_data(data, n_steps):
    return array(data[len(data) - n_steps:].astype('float32'))


def predict_future(model, data, predict_steps, n_steps, n_input, empty_data_index, end_prediction_row):
    result = list()
    new_data = data[:empty_data_index]
    for i in range(predict_steps):
        fit_data = prepare_future_data(new_data, n_steps)
        x_input = fit_data.reshape((1, n_steps, n_input))
        yhat = model.predict(x_input, verbose=0)
        result.append(yhat[0])

        if end_prediction_row != n_input:
            known_data = data[empty_data_index + i][end_prediction_row:]
            known_data = known_data.reshape((1, known_data.shape[0]))
            yhat = array(hstack([yhat, known_data]))

        new_data = np.concatenate([new_data, yhat])
        empty_data_index += 1
    return result


def add_empty_cols(df, cols_volume):
    df = df.T
    last_year = df.columns.values[-1]
    for i in range(1, cols_volume + 1):
        df[str(last_year + i)] = pd.Series(dtype=float)
    return df.T


def get_new_col_index(current):
    return current * 2 + 1


def add_cols(df, new_data, new_cols_vol, reconciliation_index, empty_data_index, end_prediction_row, need_add_empty_cols_to_df):
    empty_cells = empty_data_index - reconciliation_index
    if need_add_empty_cols_to_df:
        df = add_empty_cols(df, new_cols_vol)
    for col_index in range(end_prediction_row):
        title = df[col_index].values[0]
        col_data = [f"{title} прогноз", *[""] * empty_cells, *[str(round_val(v)).replace('.', ',') for v in new_data[title]]]
        if not need_add_empty_cols_to_df:
            col_data = [*col_data, *[""] * (df.shape[0] - len(col_data))]
        new_index = get_new_col_index(col_index)
        df.insert(new_index, f"{col_index}*", col_data)
    return df


def read_from_xls():
    lists = {
        "Растениеводство_без_сцен": {"header_row": 6, "end_prediction_row": 4},
        "Растениеводство_со_сцен": {"header_row": 9, "end_prediction_row": 4},
        "Платные услуги_без_сцен": {"header_row": 7, "end_prediction_row": 5},
        "Платные услуги_со_сцен": {"header_row": 10, "end_prediction_row": 5}
    }
    dataset = pd.ExcelFile('../data/Ограниченная_выборка_для_построения_моделей_средствами_Python.xlsx')
    for list_name, value in lists.items():
        yield list_name, dataset.parse(list_name, header=value["header_row"]).T, value["end_prediction_row"]


def write_to_csv(df, filename):
    print(f"Writing to {filename}")
    filepath = Path(f'../data/{filename}.csv')
    df.T.to_csv(filepath, encoding='windows-1251', index=False, sep=";")


def run():
    try:
        reconciliation_index = 4
        predict_steps = 6
        n_steps = 9
        for list_name, df, end_prediction_row in read_from_xls():
            title, train, empty_data_index = split_dataset(df.values)
            X, y = split_sequences(train, n_steps, end_prediction_row, empty_data_index)
            n_input = X.shape[2]
            model = get_model(n_steps, n_input, end_prediction_row, X, y)
            predict = predict_exists(model, train, reconciliation_index, n_steps, n_input, empty_data_index)
            future = predict_future(model, train, predict_steps, n_steps, n_input, empty_data_index, end_prediction_row)
            result = dict()
            for ep in range(end_prediction_row):
                row_result = []
                for ri in range(reconciliation_index):
                    row_result.append(predict[ri][ep])
                for ps in range(predict_steps):
                    row_result.append(future[ps][ep])
                result[title[ep]] = row_result
            df = add_cols(df, result, predict_steps, reconciliation_index, empty_data_index, end_prediction_row, end_prediction_row == n_input)
            write_to_csv(df, list_name)
    except Exception as e:
        ax = 1
        print(list_name)
        raise e


if __name__ == '__main__':
    run()
