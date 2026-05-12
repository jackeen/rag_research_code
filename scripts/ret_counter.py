import pandas as pd

from tools.data_loader import get_no_tail_csv_log_path

if __name__ == "__main__":
    md_ret = "x/88"
    f_ret = "x/8"
    md_list = pd.read_csv(get_no_tail_csv_log_path(md_ret))["similarity"].to_list()
    f_list = pd.read_csv(get_no_tail_csv_log_path(f_ret))["similarity"].to_list()

    bigger = 0
    equal = 0
    smaller = 0
    for i, md_score in enumerate(md_list):
        md_score_f = float(md_score)
        f_score_f = float(f_list[i])
        if f_score_f > md_score_f:
            bigger += 1
        if f_score_f < md_score_f:
            smaller += 1
        if f_score_f == md_score_f:
            equal += 1

    print(f"bigger: {bigger}, equal: {equal}, smaller: {smaller}")
