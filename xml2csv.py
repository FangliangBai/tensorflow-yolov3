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
Parameters:
1. Path to XML files
XML_PATH = '.../Annotation/', which contains the XML files.
IMG_PATH = '.../JPEGImage/', which contains the images.
2. Path to class class.names files
NAMES = '.../sixray.names', which contains the class name.
3. Output path
OUTPUT_PATH = '.../tensorflow-yolov3/'

Output:
Generate train and test tables under the path specified by OUTPUT_PATH:
..output_path/train.txt
..output_path/test.txt

Output csv format:
xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20
xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id
"""
import os
import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

XML_PATH = '/media/kent/DISK2/SBRI_Project/dataset_mmwave/file to train_yolo/Annotation/'
IMG_PATH = '/media/kent/DISK2/SBRI_Project/dataset_mmwave/file to train_yolo/Image/'
NAMES = '/media/kent/DISK2/SBRI_Project/dataset_mmwave/file to train_yolo/mm-wave.names'
OUTPUT_PATH = '/media/kent/DISK2/SBRI_Project/dataset_mmwave/file to train_yolo/'
OUTPUT_TRAIN_TBLNAME = 'mm-wave_train.txt'
OUTPUT_TEST_TBLNAME = 'mm-wave_test.txt'


def xml_to_csv(path):
    xml_list = []
    cont = 1
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.findall('object') == []:
            print("[Warning] {} contains no <object> section.".format(xml_file))
            continue
        line = []
        img_path = [str(IMG_PATH + root.find('filename').text)]
        line += img_path
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
                # The classes for mm-wave dataset is
                # 0 scissors, 1 knife, 2 water, 3 oil, 4 cardboard.
                # However, cardboard is not considered, so we skip it in the following if block.
                if value[-1] == 4:
                    continue
                value = np.array(value, dtype=int).tolist()
                line += [value, '_']
            else:
                print("[Warning] {} contains incomplete <object> section.".format(xml_file))
                continue
        xml_list.append(line)
        # print(cont)
        cont += 1
        
    xml_list = np.array(xml_list)
    xml_df = pd.DataFrame(xml_list)

    xml = xml_df.values
    xml_train, xml_test = train_test_split(xml, test_size=0.2)

    xml_train_df = pd.DataFrame(xml_train)
    xml_test_df = pd.DataFrame(xml_test)

    xml_train_df.to_csv((OUTPUT_PATH + OUTPUT_TRAIN_TBLNAME), index=False, header=False, sep=' ', )
    xml_test_df.to_csv((OUTPUT_PATH + OUTPUT_TEST_TBLNAME), index=False, header=False, sep=' ', )
    

def class_name2index(name, names):
    """
    convert class name to corresponding index number.
    :param name:
    :param names:
    :return:
    """
    name_list = pd.read_csv(names, header=None, index_col=None, dtype='str')
    name_list = name_list.values
    name_list = np.reshape(name_list, [-1])
    print('[Error] The xml class name', name, 'can not match the pre-defined names.')
    index = np.where(name_list == name)
    return index[0][0]


def refine_txt(file):
    """
    remove the unwanted symbols.
    :param file:
    :return:
    """
    s = open(file).read()
    s = s.replace('[', '')
    s = s.replace("]", '')
    s = s.replace('"', '')
    s = s.replace("'", '')
    s = s.replace('jpg,', 'jpg')
    s = s.replace(', ', ',')
    s = s.replace(',_,', ' ')
    s = s.replace(',_', '')
    
    f = open(file, 'w')
    f.write(s)
    f.close()


def main():
    xml_to_csv(XML_PATH)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()
    refine_txt((OUTPUT_PATH + OUTPUT_TRAIN_TBLNAME))
    refine_txt((OUTPUT_PATH + OUTPUT_TEST_TBLNAME))
