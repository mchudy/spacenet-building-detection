#!/usr/bin/env bash

declare -A datasets=(
    ["rio"]="AOI_1_Rio"
    ["vegas"]="AOI_2_Vegas"
    ["paris"]="AOI_3_Paris"
    ["paris2"]="AOI_3_Paris2"
    ["shanghai"]="AOI_4_Shanghai"
    ["khaortum"]="AOI_5_Khartoum"
)

city="$1"
dataset="${datasets[$city]}"
basedir=$(dirname $0)

echo "Downloading $dataset data from AWS"
mkdir -p ./data/$city

if [ "$city" == "rio" ]; then
    tarfile="./data/$city/processedBuildingLabels.tar.gz"
    tempfolder="./data/$city/processedBuildingLabels"

    python $basedir/../src/download_data.py -np 2 -s 100 s3://spacenet-dataset/$dataset/processedData/processedBuildingLabels.tar.gz ./data/$city
    #aws s3api get-object --bucket spacenet-dataset --key ${dataset}/processedData/processedBuildingLabels.tar.gz  --request-payer requester $tarfile

    tar xzf $tarfile -C ./data/$city
    mv $tempfolder/* ./data/$city
    rmdir $tempfolder
    tar xzf ./data/$city/3band.tar.gz -C ./data/$city
    tar xzf ./data/$city/8band.tar.gz -C ./data/$city
    tar xzf ./data/$city/vectordata/geojson.tar.gz -C ./data/$city/vectordata
    tar xzf ./data/$city/vectordata/summarydata.tar.gz -C ./data/$city/vectordata
    rm ./data/$city/*.tar.gz
    rm ./data/$city/vectordata/*.tar.gz
else
    tarfile="./data/$city/$dataset_Train.tar.gz"
    tempfolder="./data/$city/$dataset_Train"

    python $basedir/../src/download_data.py -np 2 -s 100 s3://spacenet-dataset/$dataset/${dataset}_Train.tar.gz ./data/$city
    #aws s3api get-object --bucket spacenet-dataset --key ${dataset}/${dataset}_Train.tar.gz  --request-payer requester $tarfile

    tar xzf $tarfile -C ./data/$city
    mv $tempfolder/* ./data/$city
    rmdir $tempfolder
    rm $tarfile
fi

echo "Download completed"
