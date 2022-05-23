import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from numpy import array

from src.utils import round_val, split_dataframe, write_to_csv, add_cols, read_from_xls, prepare_future_data, \
    prepare_reconciliation_data


def split_sequence(sequence, n_steps, empty_column_index):
    """
    Split the sequence into training blocks and reconciliation blocks.
    The size of the training block corresponds to n_steps
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1 or end_ix >= empty_column_index:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def get_model(n_steps, n_input, train_x, train_y):
    """
    Train and get a Vanilla LSTM recurrent neural network model
    """
    model = Sequential()
    model.add(LSTM(150, activation='relu', input_shape=(n_steps, n_input)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_x, train_y, epochs=3000, verbose=0)
    return model


def predict_reconciliation(model, data, reconciliation_number, n_steps, n_input, empty_column_index):
    """
    Starting with len(data) - reconciliation_number generates one block each
    for input into the neural network until it reaches len(data). The received
    forecasts are added to an array, which is later added to a separate data row
    """
    predict = list()
    for fit_data in prepare_reconciliation_data(data, reconciliation_number, n_steps, empty_column_index):
        x_input = fit_data.reshape((1, n_steps, n_input))
        yhat = model.predict(x_input, verbose=0)[0][0]
        predict.append(round_val(yhat))
    return predict


def predict_future(model, data, predict_steps, n_steps, n_input, empty_column_index):
    """
    Prepares the block for input into the neural network,
    receives the forecast and adds the received values to the new dataframe.
    The number of predictions made corresponds to predict_steps
    """
    result = list()
    new_data = data[:empty_column_index]
    for _ in range(predict_steps):
        fit_data = prepare_future_data(new_data, n_steps)
        x_input = fit_data.reshape((1, n_steps, n_input))
        yhat = model.predict(x_input, verbose=0)[0][0]
        result.append(round_val(yhat))
        new_data = np.concatenate([new_data, array([yhat])])
        empty_column_index += 1
    return result


def process_lstm(df, n_steps, index_last_predicted_row, reconciliation_number, predict_steps):
    # number of rows to input
    n_input = 1

    title, train, empty_column_index = split_dataframe(df.values)
    result = dict()
    for row_index in range(index_last_predicted_row):
        row = train[:, row_index]
        X, y = split_sequence(row, n_steps, empty_column_index)
        X = X.reshape((X.shape[0], X.shape[1], n_input))
        model = get_model(n_steps, n_input, X, y)
        predict = predict_reconciliation(model, row, reconciliation_number, n_steps, n_input, empty_column_index)
        future = predict_future(model, row, predict_steps, n_steps, n_input, empty_column_index)
        result[title[row_index]] = [*predict, *future]
        print(f"{title[row_index]} predicted")

    # do we need to add new empty columns to the end of dataframe
    need_add_empty_cols_to_df = empty_column_index + predict_steps > df.shape[0] - 1

    return add_cols(df, result, predict_steps, reconciliation_number, empty_column_index, index_last_predicted_row,
                    need_add_empty_cols_to_df)


def run_lstm(reconciliation_number, predict_steps):
    # number of values for input
    n_steps = 9

    for list_name, df, index_last_predicted_row in read_from_xls():
        result = process_lstm(df, n_steps, index_last_predicted_row, reconciliation_number, predict_steps)
        write_to_csv(result, list_name, "single")
