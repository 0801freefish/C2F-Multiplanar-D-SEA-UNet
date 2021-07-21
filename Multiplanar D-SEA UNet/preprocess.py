#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:35:13 2020

@author: bme123
"""

"""
Data preproccess for MUltiPlan

convert t1 t1c1 t2 fliar to one file
"""
import glob
import nibabel as nib 
import numpy as np
import os
import shutil

def trainData():
    flag1,flag2 = 0,0
    count = 0
    for i in glob.glob('/home/yzk/桌面/Neuro-data/3DUnetCNN-yu/brats/data/preprocessed/New-plaque-lr/*'):
        if count<1000:
            # print(i,count)
            for j in glob.glob(i + '/*'):
                if 'data_t1.nii.gz' in j:
                    shutil.copy(j, './data_folder-plaque-lr/train/images/' + str(j.split('/')[-2]) + '.nii.gz')
                    flag1 = 1
                if 'label_plaque.nii.gz' in j:
                    shutil.copy(j, './data_folder-plaque-lr/train/labels/' + str(j.split('/')[-2]) + '.nii.gz')

                    flag2 = 1
                    #print(j)

                if flag1 == 1 and flag2 == 1:
                    count = count + 1
                    flag1 = 0
                    flag2 = 0
        else:
            print(count)
            break


    
def testData():
    flag1,flag2 = 0,0
    count = 0
    for i in glob.glob('/home/yzk/桌面/Neuro-data/3DUnetCNN-yu/brats/data/preprocessed/New-fullcrop-test-pred-resize/*'):
        #print(i,count)
        if count<1000:
            for j in glob.glob(i + '/*'):
                # print(j)
                if 'data_t1.nii.gz' in j:
                    shutil.copy(j, './data_folder-pred2-aug/test/images/' + str(j.split('/')[-2]) + '.nii.gz')
                    flag1 = 1
                if 'label_seg.nii.gz' in j:
                    shutil.copy(j, './data_folder-pred2-aug/test/labels/' + str(j.split('/')[-2]) + '.nii.gz')

                    flag2 = 1
                    print(j)
                if flag1 == 1 and flag2 == 1:
                    count = count + 1
                    flag1 = 0
                    flag2 = 0

#trainData()
testData()
