#!/usr/bin/env bash

BASE_PATH=$1
SOURCE_FOLDER=$2
TARGET_FOLDER=$3

echo "BASE PATH: $BASE_PATH"
echo  "Source folder: $SOURCE_FOLDER"
echo "TArget Folder: $TARGET_FOLDER"

FILES="$BASE_PATH/$SOURCE_FOLDER/*"

count=0
batch=0
for f in $FILES; do
    echo $f
    echo $count
    if [ $count -eq 5 ] ; then
        echo "Upping batch"
        batch=$((batch+1))
        echo $batch
        count=0
    fi
    cp $f "$BASE_PATH/$TARGET_FOLDER/$batch/$(basename $f)"
    count=$((count+1))

done
