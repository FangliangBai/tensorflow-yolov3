#!/usr/bin/env bash

# ================ Train ================
# Gene class.names file manually.

# Generate train.txt, test.txt.
# preset the parameters
python ./xml2csv.py

# Generate train.tfrecord, train.tfrecord.
sh ./make_sixray_tfrecords.sh

# Generate anchors.txt, get prior anchors and rescale the values to the range [0,1]
python ./kmeans.py --dataset_txt ./sixray_train.txt --anchors_txt ./sixray_anchors.txt

# Get pre-trained weights
python convert_weight.py --convert

# Train the network
# preset the parameters @ #todo mark
python quick_train.py

# Track the training
tensorboard --logdir ./data

# ================ Evaluate ================
python convert_weight.py -cf ./checkpoint/yolov3_sixray.ckpt-2500 --num_classes 6 -ap ./sixray_anchors.txt --freeze
python quick_test.py
python evaluate.py