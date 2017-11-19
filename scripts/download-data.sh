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
tarfile="./data/processedBuildingLabels.tar.gz"

echo "Downloading $dataset data from AWS"
aws s3api get-object --bucket spacenet-dataset --key ${dataset}/processedData/processedBuildingLabels.tar.gz  --request-payer requester $tarfile
mkdir -p ./data/$city
tar -xzf $tarfile -C ./data/$city
rm $tarfile
echo "Download completed"
