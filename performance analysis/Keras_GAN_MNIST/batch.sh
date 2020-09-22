#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR

JOB_FILE=jobs.sh
PYTHON_CMD="python3"

NOISE_DIMs="50 100 150"
GEN_L1s="64 128 256"
GEN_L2s="32 64 128"
DISC_L1s="32 64 128"
DISC_L2s="64 128 256"

read -r -d '' JOB_FILE_HEADER << EOM
#!/bin/bash
BASEDIR=\$(dirname "\$0")
BASENAME=\$(basename "\$0")
cd \$BASEDIR
# ↓↓↓↓↓ jobs, do NOT edit ↓↓↓↓↓
EOM

function new_job_file() {
  echo "$JOB_FILE_HEADER" > $JOB_FILE

  for NOISE_DIM in $NOISE_DIMs; do
    for GEN_L1 in $GEN_L1s; do
      for GEN_L2 in $GEN_L2s; do
        for DISC_L1 in $DISC_L1s; do
          for DISC_L2 in $DISC_L2s; do
            echo "$PYTHON_CMD Keras_GAN_MNIST.py --noise_dim $NOISE_DIM --gen_l1 $GEN_L1 --gen_l2 $GEN_L2 --disc_l1 $DISC_L1 --disc_l2 $DISC_L2" >> $JOB_FILE
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
