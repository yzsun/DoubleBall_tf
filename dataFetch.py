# -*- coding: utf-8 -*-
"""
首选
http://www.cwl.gov.cn/kjxx/ssq/hmhz/index.shtml
output: 
"""

import bs4
from bs4 import BeautifulSoup
import urllib2
import numpy as np
import pandas as pd
import re
import sys 

reload(sys)
sys.setdefaultencoding('utf8')

qihao_list = []
balls_list = []
ball_list =[]
data_df = pd.DataFrame(columns=['peri','red1','red2','red3','red4','red5','red6','blue'])
                       
for page_index in np.arange(28):  #29
    print('第%s页 读取中...'%page_index)    
    
    if page_index == 0:
        html = urllib2.urlopen('http://www.cwl.gov.cn/kjxx/ssq/hmhz/index.shtml').read()
    else:
        html = urllib2.urlopen('http://www.cwl.gov.cn/kjxx/ssq/hmhz/index_%s.shtml' %page_index).read()
    soup = BeautifulSoup(html)         
    # 期
    for qihao in soup.findAll('td',{'height':'35'}):
        qihao_list.append(qihao.string.encode('utf8'))  #转为str，否则未安装bs4的linux不支持
    
    # red 
    for kjhm in soup.findAll('p',{'class':'haoma tc'}):
        for ball in kjhm.findAll('span'):               
            balls_list.append(ball.string[:2])
            if len(ball_list)==7:
                balls_list.append(ball_list)
                ball_list=[]
            
          

                    
balls_arr = np.array(balls_list,dtype=int).reshape(-1,7) 
              


data_df['peri'] = qihao_list
data_df.iloc[:,1:] = balls_arr



                      
# store
store = pd.HDFStore('data_2013001_2016109.h5')
store['data']=data_df
store.close()


d = pd.read_hdf('data_2013001_2016109.h5')
print(d)


