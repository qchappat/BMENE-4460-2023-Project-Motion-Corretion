import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Filter fastMRI dataset and generate data info csv.')
parser.add_argument(
    '--data_dir', required=True,
    help='directory that conatins multicoil_train, multicoil_val and multicoil_test')
parser.add_argument('--output_csv', required=True, help='path where to save the csv')
parser.add_argument('--debug', action='store_true', help='debug mode')
args = parser.parse_args()

ns = ' xmlns="http://www.ismrm.org/ISMRMRD" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.ismrm.org/ISMRMRD ismrmrd.xsd"'
data = list()
# Looping through all corrupted files since their extensions are ending in 'corrupted.h5'
for p in tqdm(list(Path(args.data_dir).glob('multicoil_*/*corrupted.h5'))):
    # Looking for the corresponding uncorrupted file
    tmp = dict()
    tmp['mode'] = p.parent.name
    tmp['path'] = str(p)[:-13]+'.h5'
    with h5py.File(str(p)[:-13]+'.h5') as f:
        tmp['acquisition'] = f.attrs['acquisition']
        shape = f['kspace'].shape
        tmp['slices'] = shape[0]
        tmp['receiverChannels'] = shape[1]
        tmp['height'] = shape[2]
        tmp['width'] = shape[3]
        tmp["corrupted"] = False
        txt = f['ismrmrd_header'][()].decode('utf-8')
    txt = txt.replace(ns, '')
    root = ET.fromstring(txt)
    tmp['systemFieldStrength_T'] = float(root.find('acquisitionSystemInformation/systemFieldStrength_T').text)
    corrupted_tmp = tmp.copy()
    corrupted_tmp['path'] = str(p)
    corrupted_tmp['corrupted'] = True
    data.append(pd.DataFrame.from_dict(tmp, orient='index').T)
    data.append(pd.DataFrame.from_dict(corrupted_tmp, orient='index').T)
    if args.debug:
        break
pd.concat(data).to_csv(args.output_csv, index=False)
