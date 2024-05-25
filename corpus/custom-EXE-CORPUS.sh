#! /bin/bash

## Custom dataset --- corpus preprocessing
CURRENT_DIR=$(pwd)

if [ $# -ne 1 ]; then
    echo "Error: This script requires exactly one argument - the absolute path to the custom dataset."
    exit 1
fi

# Get path to the custom dataset
PATH_TO_CUSTOM_DATA="$1"

# Check if the path is absolute
if [[ "$PATH_TO_CUSTOM_DATA" != /* ]]; then
    echo "Error: $PATH_TO_CUSTOM_DATA is not an absolute path."
    exit 1
fi

# Check if the directory exists
if [ ! -d "$PATH_TO_CUSTOM_DATA" ]; then
    echo "Error: Directory $PATH_TO_CUSTOM_DATA does not exist."
    exit 1
fi

# Check if the directory contains at least one .wav file
if [ $(ls "$PATH_TO_CUSTOM_DATA"/*.wav 2> /dev/null | wc -l) -eq 0 ]; then
    echo "Error: No .wav files found in directory $PATH_TO_CUSTOM_DATA."
    exit 1
fi

# Extract name of the data directory
DATA_DIR="${PATH_TO_CUSTOM_DATA##*/}"

# 1. get audio files for the corpus to run inference on
mkdir -p $CURRENT_DIR/corpus/$DATA_DIR/wav
#cp "$PATH_TO_CUSTOM_DATA"/*.wav $CURRENT_DIR/corpus/$DATA_DIR/wav/
# replace hard copies with symlinks
for file in "$PATH_TO_CUSTOM_DATA"/*.wav; do
    base=$(basename "$file")
    ln -s "$file" "$CURRENT_DIR/corpus/$DATA_DIR/wav/$base"
done

# 4. convert wav to log-mel spectrogram
mkdir -p $CURRENT_DIR/corpus/$DATA_DIR/feature
python3 $CURRENT_DIR/corpus/custom_conv_wav2fe.py -d_wav $CURRENT_DIR/corpus/$DATA_DIR/wav -d_feature $CURRENT_DIR/corpus/$DATA_DIR/feature -config $CURRENT_DIR/corpus/config.json

# 8. make dataset
python3 $CURRENT_DIR/corpus/custom_make_dataset.py -f_config_in $CURRENT_DIR/corpus/config.json -f_config_out $CURRENT_DIR/corpus/$DATA_DIR/config.json
