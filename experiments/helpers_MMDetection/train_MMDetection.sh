#!/bin/bash
source activate mm
if [ -z "$1" ]
  then
    echo "Run path missing."
    exit 1
fi
echo "Run directory: $1"

if [ -z "$2" ]
  then
    echo "Dataset name missing."
    exit 1
fi
echo "Dataset name: $2"

if [ -z "$3" ]
  then
    GPU=0
  else
    GPU=$3
fi
echo "GPU IDs: $GPU"

CONFIG_PATH=$(realpath "${1}/config.py")
if [ -z "$CONFIG_PATH" ]
  then
    exit 1
fi

python - << EOF
import os
f = open(os.path.abspath("$1/config.py"), 'r+')
content = f.read()
f.seek(0, 0)
f.write(f'run_path = "{os.path.abspath("$1")}"\n')
f.write(f'run_name = "{os.path.basename(os.path.abspath("$1"))}"\n')
f.write(f'experiment_path = "{os.path.abspath(os.path.join(os.path.abspath("$1"), "../.."))}"\n')
f.write(f'experiment_name = "{os.path.basename(os.path.abspath(os.path.join(os.path.abspath("$1"), "../..")))}"\n')
f.write(f'dataset_name = "$2"\n')
f.write(content)

f.close()

EOF

python /home/amvernet/mmdetection/tools/train.py $CONFIG_PATH --work-dir $1 --gpu-ids $GPU
