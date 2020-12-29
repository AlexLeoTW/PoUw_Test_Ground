#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR

JOB_FILE=jobs.sh
PYTHON_CMD="python3"

LAYER_1=("64" "128" "256")
LAYER_2=(" " "16")
IFS=""

read -r -d '' JOB_FILE_HEADER << EOM
#!/bin/bash
BASEDIR=\$(dirname "\$0")
BASENAME=\$(basename "\$0")
cd \$BASEDIR
# ↓↓↓↓↓ jobs, do NOT edit ↓↓↓↓↓
EOM

function new_job_file() {
  echo "$JOB_FILE_HEADER" > $JOB_FILE

  for layer_1 in ${LAYER_1[*]}; do
    for layer_2 in ${LAYER_2[*]}; do
      echo "$PYTHON_CMD Keras_MLP_FMNIST.py -hidden ${layer_1} ${layer_2}" >> $JOB_FILE
      echo "sed --in-place '6,7d' \$BASENAME" >> $JOB_FILE
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
