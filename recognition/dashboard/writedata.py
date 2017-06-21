# -*- coding: utf-8 -*-
"""
Created on Mon May 29 20:49:07 2017

@author: Richard
"""

import sys
import os 
import numpy as np
# Working with excel sheets
import matplotlib.pyplot as plt
import pandas as pd
import DataWarehouse as dwh
import xlsxwriter
import pickle


wd = os.getcwd()

print(wd)

list = pickle.load( open( wd+'/ecb_estoxx_50.p', "rb" ) )

x_es=pd.DataFrame(list.data,index=pd.date_range('2011-01-01 09:30:00','2014-12-31 09:30:00',freq='1D'))
x_es=x_es[np.isfinite(x_es['price'])]
x_es=x_es[['price']]

writer = pd.ExcelWriter(wd+'/es_data.xlsx')
x_es.to_excel(writer,'Sheet1')
writer.save()

list1 = pickle.load( open( wd+'/ecb_estoxx_banks.p', "rb" ) )
x_bank=pd.DataFrame(list1.data,index=pd.date_range('2011-01-01 09:30:00','2014-12-31 09:30:00',freq='1D'))
x_bank=x_bank[np.isfinite(x_bank['price'])]
x_bank=x_bank[['price']]


writer = pd.ExcelWriter(wd+'/bank_data.xlsx')
x_bank.to_excel(writer,'Sheet1')
writer.save()
