import pandas as pd
import csv


def load_data(fname):
    if fname[-4:] == ".csv":
        fname = fname[:-4]
    data = pd.read_csv("{}.csv".format(fname), error_bad_lines=False, quoting=csv.QUOTE_NONE)
    print(data.head(20))
    return data


if __name__ == "__main__":
    load_data("clean_trump.csv")
