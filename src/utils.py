from pathlib import Path
import pandas as pd
import numpy as np


def round_val(val):
    return round(float(val), 4)


def split_dataset(data):
    title, dataset = data[0], data[1:].astype('float32')
    for col in range(dataset.shape[0]):
        for row in range(dataset.shape[1]):
            if np.isnan(dataset[col][row]):
                return title, dataset, col
    return title, dataset, len(dataset)


def read_from_xls():
    lists = {
        "Растениеводство_без_сцен": {"header_row": 6, "index_last_predicted_row": 4},
        "Растениеводство_со_сцен": {"header_row": 9, "index_last_predicted_row": 4},
        "Платные услуги_без_сцен": {"header_row": 7, "index_last_predicted_row": 5},
        "Платные услуги_со_сцен": {"header_row": 10, "index_last_predicted_row": 5}
    }
    dataset = pd.ExcelFile('../data/Ограниченная_выборка_для_построения_моделей_средствами_Python.xlsx')
    for list_name, value in lists.items():
        yield list_name, dataset.parse(list_name, header=value["header_row"]).T, value["index_last_predicted_row"]


def write_to_csv(df, filename, system):
    print(f"Writing to {filename}")
    filepath = Path(f'../data/{filename}-{system}.csv')
    df.T.to_csv(filepath, encoding='windows-1251', index=False, sep=";")


def get_new_col_index(current):
    return current * 2 + 1


def add_empty_cols(df, cols_volume):
    df = df.T
    last_year = df.columns.values[-1]
    for i in range(1, cols_volume + 1):
        df[str(last_year + i)] = pd.Series(dtype=float)
    return df.T


def add_cols(df, new_data, new_cols_vol, reconciliation_number, empty_column_index, index_last_predicted_row,
             need_add_empty_cols_to_df):
    empty_cells = empty_column_index - reconciliation_number
    if need_add_empty_cols_to_df:
        df = add_empty_cols(df, new_cols_vol)
    for col_index in range(index_last_predicted_row):
        title = df[col_index].values[0]
        col_data = [f"{title} прогноз", *[""] * empty_cells,
                    *[str(round_val(v)).replace('.', ',') for v in new_data[title]]]
        if not need_add_empty_cols_to_df:
            col_data = [*col_data, *[""] * (df.shape[0] - len(col_data))]
        new_index = get_new_col_index(col_index)
        df.insert(new_index, f"{col_index}*", col_data)
    return df


def prepare_reconciliation_data(data, split_index, n_steps, empty_column_index):
    for index in range(empty_column_index - split_index, empty_column_index):
        yield np.array(data[index - n_steps: index].astype('float32'))


def prepare_future_data(data, n_steps):
    return np.array(data[len(data) - n_steps:].astype('float32'))
