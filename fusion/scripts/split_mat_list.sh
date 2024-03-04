#!/bin/bash
OUTPUT_FILE_FORMAT=mat_list
INPUT=$1
OUTPUT_FOLDER=$2
NUM_FILES=10
LINES_NUM=$(wc -l $INPUT | awk '{print $1}')
echo "TEST"
LINE_PER_FILE=$(( ($LINES_NUM+$NUM_FILES - 1) / $NUM_FILES ))
echo "TEST"
mkdir $OUTPUT_FOLDER/mat_lists
split -l $LINE_PER_FILE -d -a 1 $INPUT $OUTPUT_FOLDER/mat_lists/$OUTPUT_FILE_FORMAT --additional-suffix=.txt

echo "done :)"

exit