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
只需修改三处, 第一第二处改成对应的文件夹目录. 第三处改成对应的文件名, 这里是train.csv
1. os.chdir('D:\\python3\\models-master\\research\\object_detection\\images\\train')
2. path = 'D:\\python3\\models-master\\research\\object_detection\\images\\train'
3. output = 'train.csv'

Output csv format:
xxx/xxx.jpg 18.19 6.32 424.13 421.83 20 323.86 2.65 640.0 421.94 20
xxx/xxx.jpg 55.38 132.63 519.84 380.4 16
# image_path x_min y_min x_max y_max class_id x_min y_min ... class_id
"""
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

os.chdir('/media/kent/DISK2/SBRI_Project/dataset_sixray/SIXray-master/ks3util-1.1.1-upload/Annotation/')
path = '/media/kent/DISK2/SBRI_Project/dataset_sixray/SIXray-master/ks3util-1.1.1-upload/Annotation/'
output = 'train.csv'


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        item = []
        tree = ET.parse(xml_file)
        root = tree.getroot()
        img_path = [str(path + root.find('filename').text)]
        item += img_path
        # print(img_path)
        for member in root.findall('object'):
            if len(member) == 5:
                value = (
                    # int(root.find('size')[0].text),
                    # int(root.find('size')[1].text),
                    member[4][0].text,
                    member[4][1].text,
                    member[4][2].text,
                    member[4][3].text,
                    member[0].text,
                )
                item += value
            else:
                print("[Warning] {} contains invalid <object> section.".format(xml_file))
                continue
        xml_list.append(item)
    xml_df = pd.DataFrame(xml_list)
    return xml_df


def main():
    xml_df = xml_to_csv(path)
    xml_df.to_csv(output, index=None)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()
