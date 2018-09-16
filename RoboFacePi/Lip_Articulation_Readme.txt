The Lip Articulation for roboface was implemented by Athanasios Raptakis 
and Viacheslav Honcharenko during WS/2017. 

In order to use the Lip Articulation functionality you need to include the following code 
and the correct dependencies to your code. Then you use say() function which starts the threads etc.

As an example you can see the 


import numpy as np
from threading import Thread, Event
import face
from time import sleep, time
import os
from scipy.io import wavfilerenko
from scipy.ndimage.filters import maximum_filter1d,gaussian_filter
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
import string

#The Lip trajectory is generated
def Undersampled_Lip_Tragectory(phrase,Sleep_Time):
    A ="espeak -z -s 100 -v female5 -w test.wav "
    A=A + "'" + phrase + "'"
    #os.system("espeak -z -s 80 -v female5 -w test.wav 'Hey, why no one is looking at me? I feel neglected. I feel it! I am afraid!' ")
    os.system(A)
    samplerate, data = wavfile.read('test.wav')
    dt=1/float(samplerate)
    times = np.arange(len(data))/float(samplerate)
    N=len(times)
    max_data=maximum_filter1d(data,size=1000)
    max_data=gaussian_filter(max_data,sigma=100)
    max_Amplitude=10
    Amplitude=max_Amplitude*(max_data/float(np.max(max_data)))
    n=Sleep_Time*samplerate
    Amp=[]
    T=[]
    i=0
    while (i*n<N):
        Amp.append(Amplitude[int(i*n)])
        T.append(times[int(i*n)])
        i=i+1
    Amp=np.array(Amp)
    T=np.array(T)
    '''
    plt.figure(1)
    plt.suptitle(phrase)
    plt.subplot(211)
    plt.plot(times,data)
    plt.plot(times,max_data,'r')
    plt.subplot(212)
    plt.plot(times,Amplitude)
    plt.plot(T,Amp,'r*')
    plt.show()
    '''
    return Amp,T


# Thread that moves Lips
def MoveLips(Sleep_Time, Amplitude, flag):
    roboFace.setSpeedLips(127)
    i=0
    while flag.isSet() and i < len(Amplitude):
        roboFace.moveLips(int(Amplitude[i]))
        sleep(Sleep_Time)
        i = i + 1
    
    if ~flag.isSet():
        roboFace.moveLips(0)
        sleep(0.05)
    

#Thread That creates sound
def Talk(phrase, flag):
    A = "espeak -z -s 100 -v female5 "
    A = A + "'" + phrase + "'"
    os.system(A)
    flag.clear()

#Say function which starts the two parallel threads
def Say(text):
    phrases=sent_tokenize(text)
    for phrase in phrases:
            phrase=phrase.replace("'"," ")
	    flag = Event()
	    flag.set()
            Sleep_Time=0.05
	    Amplitude,Time=Undersampled_Lip_Tragectory(phrase,Sleep_Time)
	    

	    thread_movement = Thread(target=MoveLips, args=(Sleep_Time, Amplitude, flag))
	    thread_talk = Thread(target=Talk, args=(phrase, flag))
	    thread_talk.start()
	    thread_movement.start()
	    thread_talk.join()
	    thread_movement.join()

