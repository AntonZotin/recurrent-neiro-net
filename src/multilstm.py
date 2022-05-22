import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from numpy import array
from numpy.ma import hstack

from src.utils import split_dataset, write_to_csv, add_cols, read_from_xls, round_val, prepare_future_data, \
    prepare_reconciliation_data


def split_sequences(sequences, n_steps, index_last_predicted_rows, empty_column_index):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences) - 1 or end_ix >= empty_column_index:
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :index_last_predicted_rows]
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


def predict_reconciliation(model, data, split_index, n_steps, n_input, empty_column_index):
    predict = list()
    for fit_data in prepare_reconciliation_data(data, split_index, n_steps, empty_column_index):
        x_input = fit_data.reshape((1, n_steps, n_input))
        yhat = model.predict(x_input, verbose=0)
        predict.append(yhat[0])
    return predict


def predict_future(model, data, predict_steps, n_steps, n_input, empty_column_index, index_last_predicted_row):
    result = list()
    new_data = data[:empty_column_index]
    for i in range(predict_steps):
        fit_data = prepare_future_data(new_data, n_steps)
        x_input = fit_data.reshape((1, n_steps, n_input))
        yhat = model.predict(x_input, verbose=0)
        result.append(yhat[0])

        if index_last_predicted_row != n_input:
            known_data = data[empty_column_index + i][index_last_predicted_row:]
            known_data = known_data.reshape((1, known_data.shape[0]))
            yhat = array(hstack([yhat, known_data]))

        new_data = np.concatenate([new_data, yhat])
        empty_column_index += 1
    return result


def process_multilstm(df, n_steps, index_last_predicted_row, reconciliation_number, predict_steps):
    title, train, empty_column_index = split_dataset(df.values)
    X, y = split_sequences(train, n_steps, index_last_predicted_row, empty_column_index)

    # number of rows to input
    n_input = X.shape[2]
    model = get_model(n_steps, n_input, index_last_predicted_row, X, y)
    predict = predict_reconciliation(model, train, reconciliation_number, n_steps, n_input, empty_column_index)
    future = predict_future(model, train, predict_steps, n_steps, n_input, empty_column_index, index_last_predicted_row)
    result = dict()
    for row_index in range(index_last_predicted_row):
        row_result = []
        for ri in range(reconciliation_number):
            row_result.append(round_val(predict[ri][row_index]))
        for ps in range(predict_steps):
            row_result.append(round_val(future[ps][row_index]))
        result[title[row_index]] = row_result

    need_add_empty_cols_to_df = empty_column_index + predict_steps > df.shape[0] - 1
    return add_cols(df, result, predict_steps, reconciliation_number, empty_column_index, index_last_predicted_row,
                    need_add_empty_cols_to_df)


def run_multilstm(reconciliation_number, predict_steps):
    # number of values for input
    n_steps = 9
    for list_name, df, index_last_predicted_row in read_from_xls():
        result = process_multilstm(df, n_steps, index_last_predicted_row, reconciliation_number, predict_steps)
        write_to_csv(result, list_name, "multi")
