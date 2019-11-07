#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR

JOB_FILE=jobs.sh
PYTHON_CMD="python3"

CONV1_FILTERS="16 32 64"
CONV1_KERNEL_SIZE="2 3 4"
CONV2_FILTERS="16 32 64"
POOL_SIZE="2 3"
DENSE="64 128 256"

read -r -d '' JOB_FILE_HEADER << EOM
#!/bin/bash
BASEDIR=\$(dirname "\$0")
BASENAME=\$(basename "\$0")
cd \$BASEDIR
# ↓↓↓↓↓ jobs, do NOT edit ↓↓↓↓↓
EOM

function new_job_file() {
  echo "$JOB_FILE_HEADER" > $JOB_FILE

  for conv1_filters in $CONV1_FILTERS; do
    for conv1_kernel_size in $CONV1_KERNEL_SIZE; do
      for conv2_filters in $CONV2_FILTERS; do
        for pool_size in $POOL_SIZE; do
          for dense in $DENSE; do
            echo "$PYTHON_CMD Keras_CNN_MNIST.py -conv1 $conv1_filters $conv1_kernel_size -conv2 $conv2_filters -pool $pool_size -dense $dense" >> $JOB_FILE
            echo "sed --in-place '6,7d' \$BASENAME" >> $JOB_FILE
          done
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
