import numpy as np
from threading import Thread, Event
import face
from time import sleep, time
import os
from scipy.io import wavfile
from scipy.ndimage.filters import maximum_filter1d,gaussian_filter
import string
import re
#download nltk punkt in order to complete nltk set-up
#nltk.download()

# instructions HAVE to be exactly in this format for the parser to work properly.
# <emotion> phrase(s) (sleep)
# anytime you want a new emotion or sleep for some seconds, you have to write all 3 components.
# instructions = '<smile> Hi. My name is . (0) <neutral> How are you?(10) <sad> I cannot see you. (4) <smile> Ah there you are. (10)'


#The Lip trajectory is generated
def Undersampled_Lip_Tragectory(phrase,Sleep_Time):
    A ="espeak -z -s 100 -v female5 -w test.wav "
    A=A + "'" + phrase + "'"
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
#    phrases=sent_tokenize(text)
#    for phrase in phrases:
#            phrase=phrase.replace("'"," ")
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


#Create an Instance of Roboface class
roboFace = face.Face()
roboFace.setSpeedAll(80) 
roboFace.neutral()
sleep(1)
#Give a set of instructions
roboFace.happy()
Say("happy")
sleep(1)
roboFace.sad()
Say("sad")
sleep(1)
roboFace.angry()
Say("angry")
sleep(1)
roboFace.unsure()
Say("unsure")
sleep(1)
roboFace.neutral()
Say("neutral")
sleep(1)
#########
roboFace.happy(0,0)
Say("happy")
sleep(1)
roboFace.sad(0,0)
Say("sad")
sleep(1)
roboFace.angry(0,0)
Say("angry")
sleep(1)
roboFace.unsure(0,0)
Say("unsure")
sleep(1)
roboFace.neutral(0,0)
Say("neutral")
sleep(1)


