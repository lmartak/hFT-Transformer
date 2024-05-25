#! python

import argparse
import json
import pickle
import sys
import os
sys.path.append(os.getcwd())
from model import amt
from utils import find_files

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_list', help='corpus list directory')
    parser.add_argument('-d_wav', help='wav file directory (input)')
    parser.add_argument('-d_feature', help='feature file directory (output)')
    parser.add_argument('-config', help='config file')
    args = parser.parse_args()

    print('** conv_wav2fe: convert wav to feature **')
    print(' directory')
    print('  wav     (input) : '+str(args.d_wav))
    print('  feature (output): '+str(args.d_feature))
    print(' config file      : '+str(args.config))

    # read config file
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # AMT class
    AMT = amt.AMT(config, None, None)

    wav_files_list = sorted(find_files(args.d_wav, '*.wav'))
    for wav_file_path in wav_files_list:
        fname = wav_file_path.split('/')[-1].split('.')[0]
        print(fname)

        # convert wav to feature
        a_feature = AMT.wav2feature(wav_file_path)
        with open(args.d_feature.rstrip('/')+'/'+fname+'.pkl', 'wb') as f:
            pickle.dump(a_feature, f, protocol=4)

    print('** done **')
