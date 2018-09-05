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


def Emotional_Speech(instructions):
    phrases = re.split('[<>()]', instructions)
    comp = 0
    emotion_list, phrase_list, sleep_list = [], [], []
    for i in phrases:
        i = i.strip()
        if i != '':
            comp += 1
            if comp % 3 == 1:
                emotion_list.append(i)
            elif comp % 3 == 2:
                phrase_list.append(i)
            else:
                sleep_list.append(i)

    for emotion,phrase,pause_time in zip(emotion_list,phrase_list,sleep_list):
        print(emotion,phrase,pause_time) 
        if emotion=='happy':
            roboFace.happy(movelips = False)
        elif emotion=='sad':
            roboFace.sad(movelips = False)
        elif emotion=='angry':
            roboFace.angry()
        elif emotion=='unsure':
            roboFace.unsure(movelips = False)
        elif emotion=='neutral':
            roboFace.neutral(movelips = False)
        elif emotion=='moveleft':
            roboFace.moveHeadX(0)
        elif emotion=='moveright':
            roboFace.moveHeadX(950)
        else:
            print('*** ERROR ***')
            print("Invalid syntax or argument, emotions should be 'happy','sad','angry','unsure','neutral' ")
            break
        if phrase!="silence":
            Say(phrase)
        sleep(float(pause_time))


#Create an Instance of Roboface class
roboFace = face.Face()
roboFace.setSpeedAll(60) 
roboFace.neutral()
sleep(1)

#Give a set of instructions
instructions = '<happy> Hi! My name is Roboface! (0.1) <neutral> How are you? (0.3) <sad> I am very sad! (1) <angry> And angry! (0) <unsure> I am not sure if I like you (0) <moveleft> Moving head left (1) <moveright> Moving head right (1) <neutral>  silence (1)'




#Give instructions to Emotional_Speech function
Emotional_Speech(instructions)

#when Speech is Over Return to neutral position
roboFace.neutral()



