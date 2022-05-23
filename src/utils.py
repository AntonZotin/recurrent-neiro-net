from pathlib import Path
import pandas as pd
import numpy as np


def round_val(val):
    """
    Round the float value to 4 decimal places
    :param val: float or string:
    :return float:
    """
    return round(float(val), 4)


def split_dataframe(data):
    """
    Split data to title and values. Also return index of first empty column
    :param data: dataframe:
    :return string, dataframe, int:
    """
    title, dataframe = data[0], data[1:].astype('float32')
    for col in range(dataframe.shape[0]):
        for row in range(dataframe.shape[1]):
            if np.isnan(dataframe[col][row]):
                return title, dataframe, col
    return title, dataframe, len(dataframe)


def read_from_xls():
    """
    Read and parse XLS file and return current listname, dataframe and index of last row to predict
    :return string, dataframe, index of last row to predict:
    """
    lists = {
        "Растениеводство_без_сцен": {"header_row": 6, "index_last_predicted_row": 4},
        "Растениеводство_со_сцен": {"header_row": 9, "index_last_predicted_row": 4},
        "Платные услуги_без_сцен": {"header_row": 7, "index_last_predicted_row": 5},
        "Платные услуги_со_сцен": {"header_row": 10, "index_last_predicted_row": 5}
    }
    dataframe = pd.ExcelFile('./data/data.xlsx')
    for list_name, value in lists.items():
        yield list_name, dataframe.parse(list_name, header=value["header_row"]).T, value["index_last_predicted_row"]


def write_to_csv(df, filename, system):
    """
    Write dataframe to CSV file
    :param df: dataframe:
    :param filename: name of file to write:
    :param system: name of using system:
    """
    print(f"Writing to {filename}")
    filepath = Path(f'./data/{filename}-{system}.csv')
    df.T.to_csv(filepath, encoding='windows-1251', index=False, sep=";")


def get_new_col_index(current):
    """
    Get index of new adding column to insert
    :param current: int:
    :return int:
    """
    return current * 2 + 1


def add_empty_cols(df, cols_number):
    """
    Add empty cols to the end of dataframe
    :param df: dataframe:
    :param cols_number: number of empty columns to adding to dataframe:
    :return dataframe:
    """
    df = df.T
    last_year = df.columns.values[-1]
    for i in range(1, cols_number + 1):
        df[str(last_year + i)] = pd.Series(dtype=float)
    return df.T


def add_cols(df, new_data, new_cols_number, reconciliation_number, empty_column_index, index_last_predicted_row,
             need_add_empty_cols_to_df):
    """
    Add new_data to the current dataframe through each row up to index_last_predicted_row
    :param df: dataframe:
    :param new_data: array of rows for adding:
    :param new_cols_number: number of adding columns:
    :param reconciliation_number: number of last columns to be reconciled:
    :param empty_column_index: index of first empty column:
    :param index_last_predicted_row: index of last row to predict:
    :param need_add_empty_cols_to_df: do we need to add new empty columns to the end of dataframe
    :return dataframe:
    """
    empty_cells = empty_column_index - reconciliation_number
    if need_add_empty_cols_to_df:
        df = add_empty_cols(df, new_cols_number)
    for col_index in range(index_last_predicted_row):
        title = df[col_index].values[0]
        col_data = [f"{title} прогноз", *[""] * empty_cells,
                    *[str(round_val(v)).replace('.', ',') for v in new_data[title]]]
        if not need_add_empty_cols_to_df:
            col_data = [*col_data, *[""] * (df.shape[0] - len(col_data))]
        new_index = get_new_col_index(col_index)
        df.insert(new_index, f"{col_index}*", col_data)
    return df


def prepare_reconciliation_data(data, reconciliation_number, n_steps, empty_column_index):
    """
    Creates blocks for reconciliation. the number of blocks corresponds to the reconciliation_number.
    The number of values in the block corresponds to n_steps
    :param data: dataframe:
    :param reconciliation_number: number of last columns to be reconciled:
    :param n_steps: number of values for input:
    :param empty_column_index: index of first empty column:
    :return array:
    """
    for index in range(empty_column_index - reconciliation_number, empty_column_index):
        yield np.array(data[index - n_steps: index].astype('float32'))


def prepare_future_data(data, n_steps):
    """
    Creates a block for the forecast. The number of values in the block corresponds to n_steps
    :param data: dataframe:
    :param n_steps: number of values for input:
    :return array:
    """
    return np.array(data[len(data) - n_steps:].astype('float32'))
