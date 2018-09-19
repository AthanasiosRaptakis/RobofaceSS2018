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
        if emotion=='happy_lips':
            roboFace.happy(movelips = True)
        elif emotion=='sad_lips':
            roboFace.sad(movelips = True)
        elif emotion=='angry_lips':
            roboFace.angry(movelips = True)
        elif emotion=='unsure_lips':
            roboFace.unsure(movelips = True)
        elif emotion=='neutral_lips':
            roboFace.neutral(movelips = True)
        elif emotion=='moveleft':
            roboFace.moveHeadX(0)
        elif emotion=='moveright':
            roboFace.moveHeadX(950)
        elif emotion=='happy':
            roboFace.happy(movelips = False)
        elif emotion=='sad':
            roboFace.sad(movelips = False)
        elif emotion=='angry':
            roboFace.angry(movelips = False)
        elif emotion=='unsure':
            roboFace.unsure(movelips = False)
        elif emotion=='neutral':
            roboFace.neutral(movelips = False)
        else:
            print('*** ERROR ***')
            print("Invalid syntax or argument")
            break
        if phrase!="silence":
            Say(phrase)
        sleep(float(pause_time))


#Create an Instance of Roboface class
roboFace = face.Face()
roboFace.setSpeedAll(60) 
roboFace.neutral()
sleep(1)

instructions= ['<neutral_lips> Hi! (0.01)',
               '<happy_lips> My name is Roboface! Welcome to the Robotics Lab!(0.1)',
               '<moveleft> silence (1)',
               '<moveright> silence (1)',
               '<neutral_lips>My purpose is to study Human Robot interaction. I can recognise human emotions and express my fillings through verbal and non verbal comunication(0)',
               '<neutral_lips>I can express emotions like happiness(0)',
               '<happy_lips> silence (1)',
               '<happy> Anger (1)',
               '<angry_lips> silence(1)',
               '<angry> and Sadness! (1)',
               '<sad_lips> silence (1)',
               '<moveleft> silence (1)',
               '<moveright> silence (1)',
               '<neutral_lips>silence(0.5)',
               '<neutral_lips>I am not a common robot(0.1)',
               '<happy>I can think with a neural network and speak with a real human voice, though a text to speach device(0.1)',
               '<unsure>With my Computer Vision System I can distinguish between males and females! And with my new Voice I say lots of compliments to Humans!(0.1)',
               '<moveleft> silence (1)',
               '<moveright> silence (1)',
               '<neutral_lips>silence(0.5)',
               '<happy>Also I am a great actor! I think that I should be the next StarWars maskot.(0.)'
               '<unsure>Why George Lukas hasnt made me a contract yet?(0.2)',
               '<angry>May the force be with you!(0.5)',
               '<happy>Good bye!(0)'
               ]

for instr in instructions:
	Emotional_Speech(instr)
roboFace.neutral()




