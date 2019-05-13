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
- OUTPUT_PATH = '/media/kent/DISK2/tensorflow-yolov3/'

Output:
Generate two txt files under the path specified by OUTPUT_PATH:
..output_path/train.txt
..output_path/test.txt

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
from sklearn.model_selection import train_test_split

os.chdir('/media/kent/DISK2/SBRI_Project/dataset_sixray/SIXray-master/ks3util-1.1.1-upload/Annotation/')
XML_PATH = '/media/kent/DISK2/SBRI_Project/dataset_sixray/SIXray-master/ks3util-1.1.1-upload/Annotation/'
IMG_PATH = '/media/kent/DISK2/SBRI_Project/dataset_sixray/SIXray-master/ks3util-1.1.1-upload/JPEGImage/'
NAMES = '/media/kent/DISK2/tensorflow-yolov3/sixray.names'
OUTPUT_PATH = '/media/kent/DISK2/tensorflow-yolov3/'

def xml_to_csv(path):
    xml_list = []
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
                value = np.array(value, dtype=int).tolist()
                line += value
            else:
                print("[Warning] {} contains incomplete <object> section.".format(xml_file))
                continue
        xml_list.append(line)

    xml_list = np.array(xml_list)
    xml_df = pd.DataFrame(xml_list)

    xml = xml_df.values
    xml_train, xml_test = train_test_split(xml, test_size=0.2)

    xml_train_df = pd.DataFrame(xml_train)
    xml_test_df = pd.DataFrame(xml_test)

    xml_train_df.to_csv((OUTPUT_PATH + 'sixray_train.txt'), index=False, header=False, sep=' ', )
    xml_test_df.to_csv((OUTPUT_PATH + 'sixray_test.txt'), index=False, header=False, sep=' ', )
    

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
    s = s.replace(",", '')
    
    f = open(file, 'w')
    f.write(s)
    f.close()
    
def main():
    xml_to_csv(XML_PATH)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()
    refine_txt((OUTPUT_PATH + 'sixray_train.txt'))
    refine_txt((OUTPUT_PATH + 'sixray_test.txt'))
