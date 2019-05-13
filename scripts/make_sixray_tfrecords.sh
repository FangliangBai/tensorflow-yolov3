#!/usr/bin/env bash
python core/convert_tfrecord.py --dataset_txt ./sixray_train.txt --tfrecord_path_prefix ./sixray_train
python core/convert_tfrecord.py --dataset_txt ./sixray_test.txt  --tfrecord_path_prefix ./sixray_test
