# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 17:30:01 2019

@author: Ashok
"""

from FeatureGenerator import *
import pickle
import pandas as pd
from helpers import *


class TargetFeatureGenerator(FeatureGenerator):
    
    '''
        doing nothing other than returning the target variables
    '''

    def __init__(self, name='targetFeatureGenerator'):
        super(TargetFeatureGenerator, self).__init__(name)


    def process(self, df, header='train'):

        targets = df['target'].values
        outfilename_target = header+".target.pkl"
        with open(outfilename_target, "wb") as outfile:
            pickle.dump(target, outfile, -1)
        print ('targets saved in ' , outfilename_target)
        
        return targets


    def read(self, header='train'):

        filename_target = header+".target.pkl"
        with open(filename_target, "rb") as infile:
            target = pickle.load(infile)
            print ('target.shape:')
            print (target.shape)

        return target