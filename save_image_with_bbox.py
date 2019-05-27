#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : save_image_with_bbox.py
#   Author      : Fangliang
#   Created date: 2019/05/20 - 12:40
#   Description : Read bounding boxes information from xml, show them on images, save images in a folder.
#================================================================
"""
Note:
    Image files and corresponding xml file should have the SAME NAME.

Example:
    1. change the parameter in the Parameters section.
"""


import os
import xml.dom.minidom
import cv2 as cv

"""
Parameters
"""
ImgPath = '/media/kent/DISK2/SBRI_Project/dataset_mmwave/15.5.2019/'    # image folder
AnnoPath = '/media/kent/DISK2/SBRI_Project/dataset_mmwave/mylabel/'    # xml folder
save_path = '/media/kent/DISK2/SBRI_Project/dataset_mmwave/annotated_imgs/'     # annotated image folder


def draw_anchor(ImgPath, AnnoPath, save_path):
    imagelist = os.listdir(ImgPath)
    for image in imagelist:
        
        image_pre, ext = os.path.splitext(image)
        imgfile = ImgPath + image
        xmlfile = AnnoPath + image_pre + '.xml'
        # print(image)
        # 打开xml文档
        DOMTree = xml.dom.minidom.parse(xmlfile)
        # 得到文档元素对象
        collection = DOMTree.documentElement
        # 读取图片
        img = cv.imread(imgfile)
        
        filenamelist = collection.getElementsByTagName("filename")
        filename = filenamelist[0].childNodes[0].data
        print(filename)
        # 得到标签名为object的信息
        objectlist = collection.getElementsByTagName("object")
        
        for objects in objectlist:
            # 每个object中得到子标签名为name的信息
            namelist = objects.getElementsByTagName('name')
            # 通过此语句得到具体的某个name的值
            objectname = namelist[0].childNodes[0].data
            
            bndbox = objects.getElementsByTagName('bndbox')
            # print(bndbox)
            for box in bndbox:
                x1_list = box.getElementsByTagName('xmin')
                x1 = int(x1_list[0].childNodes[0].data)
                y1_list = box.getElementsByTagName('ymin')
                y1 = int(y1_list[0].childNodes[0].data)
                x2_list = box.getElementsByTagName('xmax')  # 注意坐标，看是否需要转换
                x2 = int(x2_list[0].childNodes[0].data)
                y2_list = box.getElementsByTagName('ymax')
                y2 = int(y2_list[0].childNodes[0].data)
                cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
                cv.putText(img, objectname, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                           color=(255, 128, 255), thickness=2)
                # cv.imshow('head', img)
                cv.imwrite(save_path + filename, img)  # save picture


if __name__ == "__main__":
    draw_anchor(ImgPath, AnnoPath, save_path)