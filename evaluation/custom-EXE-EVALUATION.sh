#! /bin/bash

## Custom dataset --- inference of MIDI transcription (1 mid file per wav file in custom data folder)
CURRENT_DIR=$(pwd)

if [ $# -ne 1 ]; then
    echo "Error: This script requires exactly one argument - name of the folder (located in ./corpus) containing the custom data (wav and feature)"
    exit 1
fi

# Get name of the data directory
DATA_DIR="$1"

FILE_CONFIG=$CURRENT_DIR/corpus/$DATA_DIR/config.json
DIR_WAV=$CURRENT_DIR/corpus/$DATA_DIR/wav
DIR_FEATURE=$CURRENT_DIR/corpus/$DATA_DIR/feature

# Check if the config file exists
if [ ! -f "$FILE_CONFIG" ]; then
    echo "Error: Config file $FILE_CONFIG does not exist."
    exit 1
fi

# Check if the wav directory exists
if [ ! -d "$DIR_WAV" ]; then
    echo "Error: Directory $DIR_WAV does not exist."
    exit 1
fi

# Check if the feature directory exists
if [ ! -d "$DIR_FEATURE" ]; then
    echo "Error: Directory $DIR_FEATURE does not exist."
    exit 1
fi

# Load provided checkpoint of model trained on MAESTRO-V3
DIR_CHECKPOINT=$CURRENT_DIR/checkpoint/MAESTRO-V3
FILE_CHECKPOINT=model_016_003.pkl

DIR_RESULT=$CURRENT_DIR/result/$DATA_DIR
mkdir -p $DIR_RESULT

MODE=combination
OUTPUT=2nd

# inference
python3 $CURRENT_DIR/evaluation/custom_m_inference.py -f_config $FILE_CONFIG -d_cp $DIR_CHECKPOINT -m $FILE_CHECKPOINT -d_wav $DIR_WAV -d_fe $DIR_FEATURE -d_mpe $DIR_RESULT -d_note $DIR_RESULT -calc_transcript -mode $MODE
