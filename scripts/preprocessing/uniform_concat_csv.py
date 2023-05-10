import argparse

import numpy as np
import pandas as pd


def uniform_split_df(*all_df_in: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    all_df_out = list()
    all_start_idx = np.array([0] * len(all_df_in))

    while True:
        ratios = np.array([len(df_in) - start_idx for df_in, start_idx in zip(all_df_in, all_start_idx)])
        if np.all(ratios == 0):
            break
        ratios = ratios / np.min(ratios)
        ratios = np.floor(ratios).astype(np.int64)
        all_end_idx = all_start_idx + ratios
        for df_in, start_idx, end_idx in zip(all_df_in, all_start_idx, all_end_idx):
            all_df_out.append(df_in.iloc[range(start_idx, end_idx)])
        all_start_idx = all_end_idx
    return (*all_df_out, )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_in', nargs='+', required=True)
    parser.add_argument('--csv_out', required=True)
    args = parser.parse_args()

    print('args =', args)

    all_df_in = [pd.read_csv(fname) for fname in args.csv_in]
    all_df_in = sorted(all_df_in, key=len)
    all_df_out = [df_in[df_in['mode'] == 'multicoil_train'] for df_in in all_df_in]
    all_df_out += [*uniform_split_df(*(df_in[df_in['mode'] == 'multicoil_val'] for df_in in all_df_in))]
    all_df_out += [*uniform_split_df(*(df_in[df_in['mode'] == 'multicoil_test'] for df_in in all_df_in))]
    df_out = pd.concat(all_df_out)
    df_out.to_csv(args.csv_out, index=False)
