from src.lstm import run_lstm
from src.multilstm import run_multilstm

if __name__ == '__main__':

    # number of last columns to be reconciled
    reconciliation_number = 4

    # number of times to predict
    predict_steps = 6

    # id of system: 1 - single lstm, 2 - multilstm
    system = 2

    if system == 1:
        print("Start lstm system")
        run_lstm(reconciliation_number, predict_steps)
        print("End lstm system")

    elif system == 2:
        print("Start multilstm system")
        run_multilstm(reconciliation_number, predict_steps)
        print("End multilstm system")

    else:
        raise RuntimeError(f"Unknown system {system}")
