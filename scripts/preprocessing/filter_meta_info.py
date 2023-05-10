import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description='Filter meta info csv.')
parser.add_argument('--input_csv', required=True, help='meta info csv')
parser.add_argument('--output_csv', required=True, help='output csv')
parser.add_argument(
    '--acquisition', choices=['AXFLAIR', 'AXT1', 'AXT1POST', 'AXT1PRE', 'AXT2'], help='acquisition type')
parser.add_argument('--slices', type=int, help='slices')
parser.add_argument('--receiverChannels', type=int, help='channels')
parser.add_argument('--height', type=int, help='height')
parser.add_argument('--width', type=int, help='width')
parser.add_argument('--systemFieldStrength_T', type=float, choices=[1.494, 2.8936], help='field strength')
parser.add_argument('--append', action='store_true')
args = parser.parse_args()

print(args)

df_in = pd.read_csv(args.input_csv)
index = df_in['acquisition'] == args.acquisition
index &= df_in['slices'] == args.slices
index &= df_in['receiverChannels'] == args.receiverChannels
index &= df_in['height'] == args.height
index &= df_in['width'] == args.width
index &= df_in['systemFieldStrength_T'] == args.systemFieldStrength_T
df_out = df_in[index].reset_index(drop=True).copy()
df_out = df_out.iloc[np.random.default_rng(np.iinfo(np.uint64).max - 19990902).permutation(len(df_out))]
df_out.drop(columns="mode", inplace=True)

# Spliting into uncorrupted and corrupted data
df_out_uncorrupted = df_out[df_out['corrupted'] == False].copy()
df_out_currupted = df_out[df_out['corrupted'] == True].copy()

# Spliting into train, val and test with 60% for training, 20% for validation and 20% for testing
train_num = int(0.6 * len(df_out_uncorrupted))
val_num = int(0.2 * len(df_out_uncorrupted))
test_num = len(df_out_uncorrupted) - train_num - val_num

# Assigning mode to each data
df_out_uncorrupted['mode'] = ['multicoil_train'] * train_num + ['multicoil_val'] * val_num + ['multicoil_test'] * test_num
df_out_currupted['mode'] = ['multicoil_train'] * train_num + ['multicoil_val'] * val_num + ['multicoil_test'] * test_num

# Concatenating uncorrupted and corrupted data
df_out = pd.concat([df_out_uncorrupted, df_out_currupted])

# Saving to csv
header = False if args.append else True
mode = 'a' if args.append else 'w'
df_out.to_csv(args.output_csv, header=header, index=False, mode=mode)
