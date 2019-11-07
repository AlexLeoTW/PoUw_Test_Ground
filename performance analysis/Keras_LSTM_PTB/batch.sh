#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR

JOB_FILE=jobs.sh
PYTHON_CMD="python3"

STEP_SIZE="15 30 45"
BATCH_SIZE="10 20 30"
EMBD_SIZE="200 500 800"
LSTM2_SIZE="100 300 500"

read -r -d '' JOB_FILE_HEADER << EOM
#!/bin/bash
BASEDIR=\$(dirname "\$0")
BASENAME=\$(basename "\$0")
cd \$BASEDIR
# ↓↓↓↓↓ jobs, do NOT edit ↓↓↓↓↓
EOM

function new_job_file() {
  echo "$JOB_FILE_HEADER" > $JOB_FILE

  for num_steps in $STEP_SIZE; do
    for batch_size in $BATCH_SIZE; do
      for embd_size in $EMBD_SIZE; do
        for lstm2_size in $LSTM2_SIZE; do
          echo "$PYTHON_CMD Keras_LSTM_PTB.py -t --step $num_steps --batch $batch_size --embd $embd_size --lstm2 $lstm2_size" >> $JOB_FILE
          echo "sed --in-place '6,7d' \$BASENAME" >> $JOB_FILE
        done
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
