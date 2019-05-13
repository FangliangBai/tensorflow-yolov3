#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : Pycharm
#   File name   : xml2csv.py
#   Author      : Fangliang
#   Created date: 2019-05-13
#   Link        : https://blog.csdn.net/as472780551/article/details/80645861
#
# ================================================================
"""
Input:
1. Path to XML files
os.chdir('/media/kent/DISK2/SBRI_Project/dataset_sixray/SIXray-master/ks3util-1.1.1-upload/Annotation/')
XML_PATH = '/media/kent/DISK2/SBRI_Project/dataset_sixray/SIXray-master/ks3util-1.1.1-upload/Annotation/'
2. Path to class names files
- NAMES = '/media/kent/DISK2/tensorflow-yolov3/sixray.names'
3. Output path
- OUTPUT = 'train.csv'

Output csv format:
xxx/xxx.jpg 18.19 6.32 424.13 421.83 20 323.86 2.65 640.0 421.94 20
xxx/xxx.jpg 55.38 132.63 519.84 380.4 16
# image_path x_min y_min x_max y_max class_id x_min y_min ... class_id
"""
import os
import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

os.chdir('/media/kent/DISK2/SBRI_Project/dataset_sixray/SIXray-master/ks3util-1.1.1-upload/Annotation/')
XML_PATH = '/media/kent/DISK2/SBRI_Project/dataset_sixray/SIXray-master/ks3util-1.1.1-upload/Annotation/'
NAMES = '/media/kent/DISK2/tensorflow-yolov3/sixray.names'
OUTPUT = 'train.csv'

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        item = []
        tree = ET.parse(xml_file)
        root = tree.getroot()
        img_path = [str(path + root.find('filename').text)]
        item += img_path
        for member in root.findall('object'):
            if len(member) == 5:
                value = (
                    # int(root.find('size')[0].text),
                    # int(root.find('size')[1].text),
                    round(float(member[4][0].text)),
                    round(float(member[4][1].text)),
                    round(float(member[4][2].text)),
                    round(float(member[4][3].text)),
                    class_name2index(member[0].text, NAMES),
                )
                item += value
            else:
                print("[Warning] {} contains invalid <object> section.".format(xml_file))
                continue
        xml_list.append(item)
    xml_df = pd.DataFrame(xml_list)
    return xml_df


def class_name2index(name, names):
    name_list = pd.read_csv(names, header=None, index_col=None, dtype='str')
    name_list = name_list.values
    name_list = np.reshape(name_list, [-1])
    index = np.where(name_list == name)
    return index[0][0]
    
    
def main():
    xml_df = xml_to_csv(XML_PATH)
    xml_df.to_csv(OUTPUT, index=False, header=False, sep=' ',)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()
