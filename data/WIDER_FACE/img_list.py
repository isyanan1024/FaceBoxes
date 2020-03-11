# -*- coding: utf-8 -*-
# @Author: Yan An
# @Date: 2020-02-22 20:53:15
# @Last Modified by: Yan An
# @Last Modified time: 2020-03-10 18:08:19
# @Email: an.yan@intellicloud.ai

import os
import argparse

parser = argparse.ArgumentParser('detect images')
parser.add_argument("--imgPath", type=str, default=None, help="image Path")
args = parser.parse_args()

f = open('img_list.txt', 'w')

li = [f.write(x + ' ' + x.replace('jpg', 'xml') + '\n') for x in os.listdir(args.imgPath)]