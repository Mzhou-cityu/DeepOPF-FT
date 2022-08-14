# !/usr/bin/env python
# coding: utf-8
# author: Min ZHOU
# date: December 4th, 2021

import random
import sys
sys.path.append(',/MyPackage')
#****** self-defined functions ******#
from MyPackage.load_data import load_data
from MyPackage.case_generation import case_generation
from MyPackage.train import train
from MyPackage.test_model import test_model
from MyPackage.get_exp_setup import get_exp_setup
import MyPackage.config as config

if __name__ == '__main__':
    case_generation()
    load_data()
    get_exp_setup()
    if (config.test_flag == 0):
        train()
    test_model()



