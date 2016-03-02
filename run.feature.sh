#!/bin/bash

opensmile_path=/home/ubuntu/tools/openSMILE-2.1.0/bin/linux_x64_standalone_static
speech_tools_path=/home/ubuntu/tools/speech_tools/bin
ffmpeg_path=/home/ubuntu/tools/ffmpeg-2.2.4
export PATH=$opensmile_path:$speech_tools_path:$ffmpeg_path:$PATH
export LD_LIBRARY_PATH=$ffmpeg_path/libs:$opensmile_path/lib:$LD_LIBRARY_PATH

video_path=../video   # path to the directory containing all the videos. In this example setup, we are linking all the videos to "../video"

for line in $(cat "list/all.video2"); 
do
    # key frames for sift: 160x120
    ffmpeg -y -ss 0 -i $video_path/${line}.mp4 -strict experimental -t 30 -r 15 -vf scale=160x120,setdar=4:3 video/${line}.mp4
    ffmpeg -ss 0 -i video/${line}.mp4 -t 30  -vf select="eq(pict_type\,PICT_TYPE_I)" -vsync 0 keyframes/${line}_%03d.jpg

    # key frames for cnn: 224x224 
    ffmpeg -y -ss 0 -i $video_path/${line}.mp4 -strict experimental -t 30 -r 15 -vf scale=224x224,setdar=1:1 video2/${line}.mp4
    ffmpeg -ss 0 -i video/${line}.mp4 -t 30  -vf select="eq(pict_type\,PICT_TYPE_I)" -vsync 0 keyframes2/${line}_%03d.jpg
done

# SIFT
for video in $(cat "list/all.video"); 
do
    echo ${video}
    i=0
    for line in $(ls ~/hw3/keyframes/${video}_*);
    do
        let i=i+1
        ~/hw2/scripts/extractSift ${line} > ./sift_features/${video}_$(printf "%03d" $i).sift
    done
done 

# cnn features (use FC7 layer)
./scripts/create_cnn.py frame_cnn_fc7_feat/

# put cnn features from the same video into one file    
for video in $(cat "list/all.video"); 
do
    echo ${video}
    for line in $(ls ~/hw3/frame_cnn_fc7_feat/${video}_*);
    do
        cat ${line} >> cnn_fc7_features/${video}.cnn
        echo ' ' >> cnn_fc7_features/${video}.cnn 
    done
done 

