#This short demo was created by Athanasios Raptakis and Viacheslav Honcharenko 
#during WS2017 for the  Robotics Practical Lip articulation Roboface at heidelberg uni

import numpy as np
from threading import Thread, Event
import face
from time import sleep, time
import os
from scipy.io import wavfile
from scipy.ndimage.filters import maximum_filter1d,gaussian_filter
import string

#Create an Instance of Roboface class
roboFace = face.Face(x_weight=0.8, y_weight=0.2)

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
    sleep(0.5)
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
def Say(phrase):
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


#Example - Demo of the Robot
roboFace.setSpeedHead(60)
sleep(1)
Say('Hi!')
roboFace.angry()
sleep(1)
roboFace.neutral()
Say('My name is Roboface! Welcome to the Robotics Lab!')
roboFace.moveHead(500,500)
sleep(1)
roboFace.moveHead(0,500)
sleep(1)
roboFace.neutral()
Phr1=["My purpose is to study Human Robot interaction","I can recognise human emotions and express my fillings though verbal and non verbal comunication","I can express emotions like happiness"]
for phr in Phr1:
    Say(phr)
roboFace.angry()
sleep(1)
Say('Anger')
roboFace.sad()
sleep(2)
roboFace.neutral()
Say('and Sadness')
roboFace.unsure()
sleep(1)
roboFace.neutral()
roboFace.moveHead(500,500)
sleep(1)
roboFace.moveHead(0,500)
sleep(1.5)
roboFace.neutral()
Say('I am not a common robot')
roboFace.angry()
roboFace.neutral()
Phr2=['I can think with a neural network and speak with a real human voice, though a text to speach device']
for phr in Phr2:
    Say(phr)
roboFace.angry()
roboFace.neutral()
Phr3=['With my Computer Vision System I can distinguish between males and females!','And with my new Voice I say lots of compliments to Humans!']
for phr in Phr3:
    Say(phr)
roboFace.moveHead(500,500)
sleep(1)
roboFace.moveHead(0,500)
roboFace.angry()
Phr4=['Also I am a great actor! I think that I should be the next StarWars maskot.']
for phr in Phr4:
    Say(phr)
roboFace.unsure()
Say('Why George Lukas hasnt made me a contract yet?')
roboFace.angry()
Say("May the force be with you!")
roboFace.neutral()
Say("Good bye!")










