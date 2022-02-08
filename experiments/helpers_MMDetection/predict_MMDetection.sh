#!/bin/bash
source activate mm

if [ -z "$1" ]
  then
    echo "Config path missing."
    exit 1
fi
echo "Config path: $1"

if [ -z "$2" ]
  then
    echo "Model path missing."
    exit 1
fi
echo "Model path: $2"

if [ -z "$3" ]
  then
    echo "Dataset name missing."
    exit 1
fi
echo "Dataset name: $3"

PREDICTIONS_PATH=$(dirname $1)
RUN_NAME=$(basename $PREDICTIONS_PATH)
PREDICTIONS_PATH=$(dirname "$PREDICTIONS_PATH")
PREDICTIONS_PATH=$(dirname "$PREDICTIONS_PATH")
PREDICTIONS_PATH="$PREDICTIONS_PATH/mmDetection_predictions/$RUN_NAME/$3"
OUT_PATH="$PREDICTIONS_PATH/predictions.pkl"
echo "Predictions will be saved under $PREDICTIONS_PATH/"

# --show-dir $PREDICTIONS_PATH  to save images
python /home/amvernet/mmdetection/tools/test.py $1 $2 --format-only --eval-options "jsonfile_prefix=$PREDICTIONS_PATH/predictions"
python /home/amvernet/mmdetection/tools/test.py $1 $2 --out $OUT_PATH --eval bbox segm --eval-options "classwise=True"
