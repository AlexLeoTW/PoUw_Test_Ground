#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR

JOB_FILE=jobs.sh
PYTHON_CMD="python3"

CONV_FILTERS="16 32 64"
CONV_KERNEL_SIZE="2 3 4"
CONV_NUM="2 4"
POOL_SIZE="2 3"
STACK="independent 2_in_a_row"

read -r -d '' JOB_FILE_HEADER << EOM
#!/bin/bash
BASEDIR=\$(dirname "\$0")
BASENAME=\$(basename "\$0")
cd \$BASEDIR
# ↓↓↓↓↓ jobs, do NOT edit ↓↓↓↓↓
EOM

function new_job_file() {
  echo "$JOB_FILE_HEADER" > $JOB_FILE

  for conv_filters in $CONV_FILTERS; do
    for conv_kernel_size in $CONV_KERNEL_SIZE; do
      for conv_num in $CONV_NUM; do
        for pool_size in $POOL_SIZE; do
          for stack in $STACK; do
            echo "$PYTHON_CMD Keras_CNN_CIFAR10.py --allow_growth --aug -conv $conv_filters $conv_kernel_size -conv_num $conv_num -pool $pool_size --stack "$stack >> $JOB_FILE
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
