#!/usr/bin/env python
#-*-coding:utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf

dataRaw = pd.read_hdf('data_2013001_2016109.h5')
dataOnehot = pd.read_hdf('dataOnehot_2013001_2016109.h5')
dataRaw = dataRaw.sort_values(by='peri')
dataOnehot = dataOnehot.sort_values(by='peri')

# para
timeStep = 2 
vec_size = lstm_size = 214  # 
out_size = 214
trainRatio = 0.9
maxEcho = 1000
trStart,trEnd = (-450,-10)
teStart,teEnd = (-10,-8) 

# data preparation
def dataPre_Rnn(data, timeStep, labels=False):
    dataPre_list = []    
    if labels:
        for row in range(len(data)-timeStep):
            dataPre_list.append( data.iloc[row+timeStep].as_matrix() )
    else:
        for row in range(len(data)-timeStep):
            dataPre_list.append( data.iloc[row:row+timeStep].as_matrix() )   
    return np.array(dataPre_list)
trainIndex = int(round(len(dataRaw)*trainRatio))
X_Raw = dataPre_Rnn(dataRaw,timeStep,False)
Y_Raw = dataPre_Rnn(dataRaw,timeStep,True)
X_Onehot = dataPre_Rnn(dataOnehot.iloc[:,1:],timeStep,False).astype(np.float32)
Y_Onehot = dataPre_Rnn(dataOnehot.iloc[:,1:],timeStep,True).astype(np.float32)
trX,trY = X_Onehot[trStart:trEnd,:], Y_Onehot[trStart:trEnd,:]
teX,teY = X_Onehot[teStart:teEnd,:], Y_Onehot[teStart:teEnd,:]

# input
X = tf.placeholder("float32",[None,timeStep,vec_size])
Y = tf.placeholder("float32",[None,out_size])
learnRate = tf.placeholder('float32')
# weights and bias
w1 = tf.Variable(tf.random_normal([out_size,out_size]))
b1 = tf.Variable(tf.random_normal([out_size],dtype=tf.float32))
w2 = tf.Variable(tf.random_normal([out_size,out_size]))
b2 = tf.Variable(tf.random_normal([out_size],dtype=tf.float32))

# model
def model(X,timeStep,lstm_size,W1,B1,W2,B2):
    XT = tf.transpose(X, [1, 0, 2])
    XR = tf.reshape(XT, [-1, lstm_size])
    X_split = tf.split(0, timeStep, XR)
   
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size,forget_bias=1.0)
    outputs, _states = tf.nn.rnn(lstm, X_split, dtype=tf.float32)
    out_lstm = outputs[-1]
    hid = tf.nn.sigmoid(tf.matmul(out_lstm,W1)+B1)
    out = tf.matmul(out_lstm,W2)+B2
    return out
    
# train 
out = tf.cast(model(X,timeStep,lstm_size,w1,b1,w2,b2),tf.float32)
cost = tf.reduce_mean(tf.reduce_sum(tf.abs(out-Y),1))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out,Y))
train_op = tf.train.RMSPropOptimizer(learnRate,0.9).minimize(cost)

# session run 
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for echo in range(maxEcho):
        # learnRate
        if echo <= 100:
            lr = 0.03
        elif echo <=150:
            lr = 0.01
        elif echo<=200:
            lr = 0.001
        elif echo<=300:
            lr = 0.0001
        elif echo<=400:
            lr = 0.00001
        elif echo<=500:
            lr = 0.000001
        elif echo<=600:
            lr = 0.0000001
        elif echo<=700:
            lr = 0.00000001
        elif echo<=800:
            lr = 0.000000001
        else:
            lr = 0.000000000001
        # train
        sess.run(train_op,feed_dict={X:trX,Y:trY,learnRate:lr})
        # test
        trOut,trCost = sess.run([out,cost],feed_dict={X:trX,Y:trY,learnRate:lr})
        teOut,teCost = sess.run([out,cost],feed_dict={X:teX,Y:teY,learnRate:lr})
        print('echo:',echo,'trCost',trCost,'teCost',teCost)
    
    # train set
    trOut = np.array(trOut)
    numRank1=numRank2=numRank3=numRank4=numRank5=numRank6=numNull=0
    for row in range(len(teY)):
        # int: 0 or 1   and  reverse
        outOnehot = teOut[row]
        outRaw = []
        for col in range(7):
            outRaw.append(outOnehot[col*33:(col+1)*33-1].argmax()-33*col+1)
        outRaw = np.array(outRaw)
        yRaw = Y_Raw[row+teStart,1:].astype(np.float32)     
        print('outRaw',outRaw)
        print('yRaw',yRaw)
        # 等级统计     
        bool_arr = yRaw==outRaw
        numTrue = sum(bool_arr)
        if numTrue == 7:
            numRank1 +=1
            rank = '1'
        elif numTrue == 6 and bool_arr[6]==0:
            numRank2 +=1
            rank = '2'
        elif numTrue == 6 and bool_arr[6]==1:
            numRank3 +=1
            rank = '3'
        elif numTrue == 5:
            numRank4 +=1
            rank ='4'
        elif numTrue ==4:
            numRank5 +=1
            rank = '5'
        elif numTrue<=3 and bool_arr[6]==1:
            numRank6 +=1
            rank = '6'
        else:
            numNull +=1
            rank = '0'
        print('### train '+str(row+1) +'###')
        print('peri:%s  time:%s  true:%s  rank:%s'\
             %(Y_Raw[trStart+row,0],Y_Raw[trStart+row,1],numTrue,rank))        
        print('outRaw',outRaw)
        print('yRaw',yRaw)
    print('############## train(total) ########################')
    print('PeriTol:%s ~ %s  TimeTol:%s ~ %s  Sum:%s'\
         %(Y_Raw[trStart,0],Y_Raw[trEnd-1,0],Y_Raw[trStart,1],Y_Raw[trEnd-1,1],trEnd-trStart))
    print('一等:%s,\n 二等:%s,\n 三等:%s,\n 四等:%s,\n 五等:%s,\n 六等:%s,\n 空:%s \n'\
                 %(numRank1,numRank2,numRank3,numRank4,numRank5,numRank6,numNull))

   # test set
    teOut = np.array(teOut)
    numRank1=numRank2=numRank3=numRank4=numRank5=numRank6=numNull=0
    for row in range(len(teY)):
        # int: 0 or 1   and  reverse
        outOnehot = teOut[row]
        outRaw = []
        for col in range(7):
            outRaw.append(outOnehot[col*33:(col+1)*33-1].argmax()-33*col+1)
        outRaw = np.array(outRaw)     
        yRaw = Y_Raw[row+teStart,2:9].astype(np.float32)
        # 等级统计     
        bool_arr = yRaw==outRaw
        numTrue = sum(bool_arr)
        if numTrue == 7:
           numRank1 +=1
           rank = '1'
        elif numTrue == 6 and bool_arr[6]==0:
            numRank2 +=1
            rank = '2'
        elif numTrue == 6 and bool_arr[6]==1:
            numRank3 +=1
            rank = '3'
        elif numTrue == 5:
            numRank4 +=1
            rank ='4'
        elif numTrue ==4:
            numRank5 +=1
            rank = '5'
        elif numTrue<=3 and bool_arr[6]==1:
            numRank6 +=1
            rank = '6'
        else:
            numNull +=1
            rank = '0'

        print('### test '+str(row+1) +'###')
        print('peri:%s  time:%s  true:%s  rank:%s'\
             %(Y_Raw[teStart+row,0],Y_Raw[teStart+row,1],numTrue,rank))
        print('outRaw',outRaw)
        print('yRaw',yRaw)
    print('############## test(total) ########################')
    print('PeriTol:%s ~ %s  TimeTol:%s ~ %s  Sum:%s'\
         %(Y_Raw[teStart,0],Y_Raw[teEnd,0],Y_Raw[teStart,1],Y_Raw[teEnd,1],trEnd-teStart))
    print('一等:%s,\n 二等:%s,\n 三等:%s,\n 四等:%s,\n 五等:%s,\n 六等:%s,\n 空:%s \n'\
                 %(numRank1,numRank2,numRank3,numRank4,numRank5,numRank6,numNull))







