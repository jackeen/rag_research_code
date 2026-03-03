import pandas as pd
from tools.data_loader import get_csv_data_path


if __name__ == '__main__':
    d_path = get_csv_data_path('questions_and_answers')
    df = pd.read_csv(str(d_path))
    x = df['books'].dropna().tolist()
    print(len(x), x)

