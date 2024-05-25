#! python

import argparse
import numpy as np
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f_config_in', help='config json file(input)', default='config.json')
    parser.add_argument('-f_config_out', help='config json file(output)', default='config_out.json')
    parser.add_argument('-max_value', help='max feature value', type=float, default=0.0)

    args = parser.parse_args()

    print(' config file')
    print('  input             : '+str(args.f_config_in))
    print('  output            : '+str(args.f_config_out))

    # read config file
    with open(args.f_config_in, 'r', encoding='utf-8') as f:
        config = json.load(f)

    config['input']['max_value'] = args.max_value
    if config['feature']['log_offset'] > 0.0:
        config['input']['min_value'] = np.log(config['feature']['log_offset']).astype(np.float32)
    else:
        config['input']['min_value'] = config['feature']['log_offset']

    # write config file
    config['input']['min_value'] = float(config['input']['min_value'])
    config['feature']['n_bins'] = config['feature']['mel_bins']
    with open(args.f_config_out, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4, sort_keys=False)

    print('** done **')
