#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR

JOB_FILE=jobs.sh
PYTHON_CMD="python3"

MAX_FEATURES="10000 20000 40000"
EMBD_SIZE="64 128 256"
CNN_TYPE="SimpleRNN GRU LSTM CuDNNGRU CuDNNLSTM"

read -r -d '' JOB_FILE_HEADER << EOM
#!/bin/bash
BASEDIR=\$(dirname "\$0")
BASENAME=\$(basename "\$0")
cd \$BASEDIR
# ↓↓↓↓↓ jobs, do NOT edit ↓↓↓↓↓
EOM

function new_job_file() {
  echo "$JOB_FILE_HEADER" > $JOB_FILE

  for max_features in $MAX_FEATURES; do
    for embd_size in $EMBD_SIZE; do
      for type in $CNN_TYPE; do
        echo "$PYTHON_CMD Keras_RNN_IMDB.py --type $type --feature $max_features --embd $embd_size" >> $JOB_FILE
        echo "sed --in-place '6,7d' \$BASENAME" >> $JOB_FILE
      done
    done
  done

  echo "" >> $JOB_FILE
  echo "rm \$BASENAME" >> $JOB_FILE
}

if [[ ! -f $JOB_FILE ]]; then
  printf "\n\nJob file \"${JOB_FILE}\" not exist, create one.\n\n"
  new_job_file
fi

bash $JOB_FILE
