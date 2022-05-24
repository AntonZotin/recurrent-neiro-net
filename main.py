import optparse

from src.lstm import run_lstm
from src.multilstm import run_multilstm


def parse_args():
    parser = optparse.OptionParser()
    parser.add_option("--rn", help="Reconciliation number", default=4)
    parser.add_option("--ps", help="Predict steps", default=6)
    parser.add_option("--sys", help="Using system: 1 - singlelstm, 2 - multilstm", default=2)
    args, _ = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    # number of last columns to be reconciled
    reconciliation_number = int(args.rn)
    print("Reconcilation number: %s" % reconciliation_number)

    # number of times to predict
    predict_steps = int(args.ps)
    print("Predict steps: %s" % predict_steps)

    # id of system: 1 - single lstm, 2 - multilstm
    system = int(args.sys)

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
