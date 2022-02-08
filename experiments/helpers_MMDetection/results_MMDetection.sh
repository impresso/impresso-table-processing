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
    echo "Dataset name missing."
    exit 1
fi
echo "Dataset name: $2"

PREDICTIONS_PATH=$(dirname $1)
RUN_NAME=$(basename $PREDICTIONS_PATH)
PREDICTIONS_PATH=$(dirname "$PREDICTIONS_PATH")
EXPERIMENT_PATH=$(dirname "$PREDICTIONS_PATH")
ANNOTATIONS_PATH="$EXPERIMENT_PATH/$2.json"
PREDICTIONS_PATH="$EXPERIMENT_PATH/mmDetection_predictions/$RUN_NAME"
PREDICTIONS_PKL_PATH="$PREDICTIONS_PATH/$2/predictions.pkl"
PREDICTIONS_BBOX_PATH="$PREDICTIONS_PATH/$2/predictions.bbox.json"
PREDICTIONS_SEGM_PATH="$PREDICTIONS_PATH/$2/predictions.segm.json"
RESULTS_PATH="$EXPERIMENT_PATH/mmDetection_results/$RUN_NAME/$2"

python /home/amvernet/mmdetection/tools/analysis_tools/coco_error_analysis.py $PREDICTIONS_SEGM_PATH $RESULTS_PATH --ann=$ANNOTATIONS_PATH --types='segm'
python /home/amvernet/mmdetection/tools/analysis_tools/eval_metric.py $1 $PREDICTIONS_PKL_PATH --eval segm
python /home/amvernet/mmdetection/tools/analysis_tools/analyze_results.py $1 $PREDICTIONS_PKL_PATH $RESULTS_PATH --show --topk 100
