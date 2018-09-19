"""
This program is free software created by Athanasios Raptakis and
Viacheslav Honcharenko during SS2018 Roboface robotics practical.
We expanded the work of previous semester done by Letitia Parcalabescu
in order to add Lip Articulation for Roboface.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import cv2
import numpy as np
from keras.models import load_model
from scipy.misc import imresize
from skimage.transform import resize, rotate
import math
import face
import h5py
import os, signal
from threading import Thread, Event
from time import sleep
from scipy.io import wavfile
from scipy.ndimage.filters import maximum_filter1d, gaussian_filter

IMAGE_SIZE = (128, 128)
IOD = 40.0


def imgCrop(image, cropBox, boxScale=1):
    '''
    Crop an area around the detected face (by OpenCV) in order to feed it into the prediction algorithm (NN).
    '''
    off = 90
    y = max(cropBox[1] - 3 * off, 0)
    x = max(cropBox[0] - 2 * off, 0)

    off = 50
    y = max(cropBox[1] - 3 * off, y)
    x = max(cropBox[0] - 2 * off, x)

    off = 20
    y = max(cropBox[1] - 3 * off, y)
    x = max(cropBox[0] - 2 * off, x)

    cropped = image[y:cropBox[1] + cropBox[3] + 90, x:cropBox[0] + cropBox[2] + 30]
    dims = cropped.shape

    return cropped, x, y


def rotateBound(image, angle, center):
    '''
    Rotates image. Used for image normalisation, so that the inter-ocular line is always horizontal for the NN.
    '''
    (cX, cY) = center
    (h, w) = image.shape[:2]

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def normaliseImage(image, eyes, xcrop, ycrop):
    '''
    Normalize faces usinginter-ocular distance i.o.d
    '''
    # resite, such that i.o.d is always same
    left_eye = eyes[0] + np.array([xcrop, ycrop, 0, 0])
    right_eye = eyes[1] + np.array([xcrop, ycrop, 0, 0])
    scale = IOD / np.linalg.norm(left_eye - right_eye)
    left_eye = scale * left_eye
    right_eye = scale * right_eye
    im = resize(image, (int(scale * image.shape[0]), int(scale * image.shape[1])), mode='edge')

    # rotate to keep inter ocular line horizontal
    diff = np.subtract(left_eye, right_eye)
    angle = math.atan2(diff[0], diff[1])
    im = rotate(im, -angle, center=(left_eye[0], left_eye[1]), preserve_range=True, mode='edge')

    # new resizing for making the image compatible with the trained NN.
    iod = np.linalg.norm(left_eye - right_eye)
    xmin = int(left_eye[0] - 1.6 * iod)
    xmax = int(left_eye[0] + 2 * iod)
    ymin = int(left_eye[1] - 1.3 * iod)
    ymax = int(right_eye[1] + 1.3 * iod)
    xmin = max(0, xmin)
    xmax = min(im.shape[0], xmax)
    ymin = max(0, ymin)
    ymax = min(im.shape[1], ymax)
    im = im[xmin:xmax, ymin:ymax, :]
    try:
        im = resize(im, IMAGE_SIZE, mode='edge')
    except:
        return None

    return im
#######################################################################
# Definition of Track_face: 
# Tracks only the face of the active user with MEDIANFLOW at opencv 
# and retrurns the bounding box of the face
#######################################################################
def Track_face(tracker,frame,face_cascade):
    
    bbox=None
    ok, bbox = tracker.update(frame)
    if ok==False:
	# Tracking failure
	cv2.putText(frame, "Tracking failure detected", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	print("Track_Face - Tracker Re-Initialisation")
	print("Number of Detected faces: ",len(faces))
	#calculate the rectangle with the biggest area
	#biggest area means that its the closest face to the robot
	max_area=0
	for face in faces:
		area=face[2]*face[3]
		if area>max_area:
			max_area=area
			bbox=tuple(face)
	tracker = cv2.TrackerMedianFlow_create()
        # re-initialise the tracker in case of lost target
	ok = tracker.init(frame, bbox)
        if len(faces)==0: ok=False
    return ok,bbox,tracker

# Re-initialization of Tracker after given period of time
def Re_Init_Tracker(frame,face_cascade):
    bbox=None
    cv2.putText(frame, "Re-Initialising Tracking", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,0,0),2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print("Tracker Re-Initialisation")
    print("Number of Detected faces: ",len(faces))
    max_area=0
    for face in faces:
        area=face[2]*face[3]
        if area>max_area:
            max_area=area
            bbox=tuple(face)
    tracker = cv2.TrackerMedianFlow_create()
    ok = tracker.init(frame, bbox)
    t1 = cv2.getTickCount()
    if len(faces)==0: ok=False
    return ok,t1,tracker


def detectFace(image,tracker,face_cascade,eye_cascade):
    # http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    unaltered_image = image.copy()
    eyes = None
    normalised_image = None
    # Track face in current frame
    ok,bbox,tracker=Track_face(tracker,image,face_cascade)
    #print(ok,bbox)
    if ok==True and bbox is not None:
        (x, y, w, h) = bbox
        x=int(x)
        y=int(y)
        w=int(w)
        h=int(h)
        bbox=(x, y, w, h)
        # show face bounding box on Webcam Preview
        cv2.rectangle(image, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 0, 255), 3)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        # normalise image in order to predict on it
        # croppedImage = imgCrop(image, face, boxScale=1)
        # detect eyes for Inter Oculat Distance
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 2:
            left_eye = eyes[0][0:2] + x
            right_eye = eyes[1][0:2] + y
            eyex = int((left_eye[0] + right_eye[0]) * .5)
            eyey = int((left_eye[1] + right_eye[1]) * .5)
            roboFace.moveHead(eyex, eyey)
        # suggestion: skip this frame as prediction, so return None, image
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            if len(eyes) == 2 and np.abs(eyes[0, 1] - eyes[1, 1]) < 10:
                offset1 = np.sqrt((eyes[0, 2] ** 2 + eyes[0, 3] ** 2)) * 0.5
                offset2 = np.sqrt((eyes[1, 2] ** 2 + eyes[1, 3] ** 2)) * 0.5
                real_eyes = eyes + np.array([[x + offset1, y + offset1, 0, 0], [x + offset2, y + offset2, 0, 0]])
                real_eyes = np.sort(real_eyes, axis=0)
                cropped_image, xcrop, ycrop = imgCrop(unaltered_image, bbox)
                normalised_image = normaliseImage(cropped_image, real_eyes, -xcrop, -ycrop)
        
    return normalised_image, image, tracker, ok


def mapAttributes(classes):
    '''F
    Map the output probabilities to the correpsonding names, like 'smile', etc.
    '''
    with open('../face_detection/wanted_attributes_normalised.txt', 'r') as f:
        attributes = f.read()
    attributes = attributes.strip('\n').split(' ')

    result = []
    for i, cl in enumerate(classes):
        if cl == True:
            result.append(attributes[i])
    return result


################################################################################
# Declaration of: say - Talk - MoveLips 
################################################################################

def Undersampled_Lip_Tragectory(phrase, Sleep_Time):
    A = "espeak -z -s 80 -v female5 -w test.wav "
    A = A + "'" + phrase + "'"
    os.system(A)
    samplerate, data = wavfile.read('test.wav')
    dt = 1 / float(samplerate)
    times = np.arange(len(data)) / float(samplerate)
    N = len(times)
    max_data = maximum_filter1d(data, size=1000)
    max_data = gaussian_filter(max_data, sigma=100)
    max_Amplitude = 10
    Amplitude = max_Amplitude * (max_data / float(np.max(max_data)))
    n = Sleep_Time * samplerate
    Amp = []
    T = []
    i = 0
    while (i * n < N):
        Amp.append(Amplitude[int(i * n)])
        T.append(times[int(i * n)])
        i = i + 1

    Amp = np.array(Amp)
    T = np.array(T)

    return Amp, T


def MoveLips(Sleep_Time, Amplitude, flag):
    roboFace.setSpeedLips(127)
    sleep(0.5)
    i = 0
    while flag.isSet() and i < len(Amplitude):
        roboFace.moveLips(int(Amplitude[i]))
        sleep(Sleep_Time)
        i = i + 1

    if ~flag.isSet():
        roboFace.moveLips(0)
        sleep(0.05)


def Talk(phrase, flag):
    A = "espeak -z -s 80 -v female5 "
    A = A + "'" + phrase + "'"
    os.system(A)
    flag.clear()


def say(phrase, flag):
    phrase = phrase.replace("'", " ")
    flag.set()
    Sleep_Time = 0.05
    Amplitude, Time = Undersampled_Lip_Tragectory(phrase, Sleep_Time)
    thread_movement = Thread(target=MoveLips, args=(Sleep_Time, Amplitude, flag))
    thread_talk = Thread(target=Talk, args=(phrase, flag))
    thread_talk.start()
    thread_movement.start()


################################################################################
# End of Declaration: say - Talk - MoveLips 
################################################################################


def sayDoSomething(pred_attr):
    talk = {'Smiling': 'I like it when people smile at me!',
            'Female': 'You are a female, am I right?',
            'Male': 'You are a male, am I right?',
            'Wearing_Earrings': 'You are wearing beautiful earrings today!',
            'Wearing_Lipstick': 'I see you are wearing lipstick today. Pretty!',
            'Blond_Hair': 'Nice blond hair!',
            'Eyeglasses': 'You are wearing eyeglasses!',
            'Brown_Hair': 'You have nice brown hair!',
            'Black_Hair': 'You have nice black hair!',
            'Gray_Hair': 'You must be a wise man, judging by your gray hair!',
            'Wavy_Hair': 'You have nice wavy hair!',
            'Straight_Hair': 'You have nice straight hair.'
            }

    if 'Smiling' in pred_attr:
        roboFace.happy(moveHead=False,movelips=False)
    elif 'Black_Hair' in pred_attr:
        roboFace.angry(moveHead=False,movelips=False)
    elif 'Eyeglasses' in pred_attr:
        roboFace.unsure(moveHead=False,movelips=False)
    else:
        roboFace.neutral(moveHead=False,movelips=False)

    index = np.random.randint(0, len(pred_attr))
    say(talk[pred_attr[index]], flag)


def getProbaStream(probStream, probs):
    if probStream == None:
        probStream = probs
    else:
        probStream = np.vstack((probStream, probs))
    return probStream

waiting_phrases=["Hi! Is anybody here?","Greetings human! Nice to meet you! ","My name is roboface! I am a friendly robot!","Hello! It's a pleasure to meet you!","I feel so lonely!"]

if __name__ == "__main__":
    #Roboface Constructor
    roboFace = face.Face(x_weight=0.8, y_weight=0.2)
    #################################################################
    # Set up tracker
    tracker = cv2.TrackerMedianFlow_create()
    Tracking_Period=5 # set tracking period before re-initialisation in seconds
    # Load Neural Net model and meanFace 
    meanFace = np.load('../face_detection/mean_face_normalised.npy')
    
    model = load_model('../face_detection/trained/pretrained_CelebA_normalised0203-05.h5')
    # Load Face Cascade and Eye Cascade classifiers
    face_cascade = cv2.CascadeClassifier('../face_detection/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('../face_detection/haarcascade_eye.xml')
    #################################################################
    # Set Speed for smoother movement
    roboFace.setSpeedAll(100)
    roboFace.setSpeedHead(80)
    flag = Event()
    flag.clear()
    #################################################################
    roboFace.neutral()
    # load the trained neural network

    cv2.namedWindow("Webcam Preview")
    vc = cv2.VideoCapture(0)  # 0 for built-in webcam, 1 for robot
    # Input from video file Only for debugging
    # vc = cv2.VideoCapture('/home/robotsnake/RoboFace18/vid/test.avi') 
    #vc.set(3,520)
    #vc.set(4,240)
    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    probStream = None
    saidNothing = 0
    process = None
    t1 = cv2.getTickCount()
    while rval:
        normalised_image, frame,tracker, ok = detectFace(frame,tracker,face_cascade,eye_cascade)
        # if a face is detected and the normalisation was successful, predict on it
        if normalised_image is not None:
            normalised_image = normalised_image[:, :, ::-1]
            # subtract mean face
            X_test = np.expand_dims(normalised_image, axis=0)
            X_test -= meanFace
            classes = model.predict_classes(X_test, batch_size=32, verbose=0)
            proba = model.predict_proba(X_test, batch_size=32, verbose=0)
            # pred_attr = mapAttributes((proba > 0.6)[0])
            # print( proba)
            # print(pred_attr)

            probStream = getProbaStream(probStream, proba)
            if saidNothing == 0 and probStream.shape[0] < 10:
                saidNothing += 1
                rval, frame = vc.read()
                cv2.imshow("Webcam Preview", frame)
                cv2.waitKey(1)                
            elif probStream.shape[0] > 10 and len(probStream.shape) >= 2:
                meanProbs = np.mean(probStream, axis=0)
                pred_attr = mapAttributes(meanProbs > 0.6)
                best = []
                if meanProbs[0] > meanProbs[1] and meanProbs[0] > meanProbs[4] and meanProbs[0] > meanProbs[2]:
                    best.append('Black_Hair')
                elif meanProbs[1] > meanProbs[0] and meanProbs[1] > meanProbs[4] and meanProbs[1] > meanProbs[2]:
                    best.append('Blond_Hair')
                elif meanProbs[2] > meanProbs[0] and meanProbs[2] > meanProbs[1]:
                    best.append('Brown_Hair')
                if meanProbs[9] < meanProbs[10]:
                    best.append('Straight_Hair')
                else:
                    best.append('Wavy_Hair')
                if meanProbs[3] > 0.6:
                    best.append('Eyeglasses')
                if meanProbs[8] > 0.6:
                    best.append('Smiling')
                if meanProbs[11] > 0.2:
                    best.append('Wearing_Earrings')
                if meanProbs[12] > 0.2:
                    best.append('Wearing_Lipstick')
                # if meanProbs[12] > 0.11 and meanProbs[11] > 0.11 and meanProbs[5] < 0.6:
                if meanProbs[5] < 0.25:
                    best.append('Female')
                elif meanProbs[12] < 0.11 and meanProbs[11] < 0.11 and meanProbs[5] > 0.85:
                    best.append('Male')
                print(meanProbs)
                print("BEST", best)

                # end NN stuff
                # postprocessing and reaction step
                sayDoSomething(best)
                saidNothing = 0
                while flag.isSet():
                    _, frame,tracker,ok = detectFace(frame,tracker,face_cascade,eye_cascade)
                    probStream = None
                    rval, frame = vc.read()
                    cv2.imshow("Webcam Preview", frame)
                    cv2.waitKey(1)
        elif saidNothing > 150:
            saidNothing = 0
            roboFace.sad()
            if ok==False:   
                index = np.random.randint(0, len(waiting_phrases))
                say(waiting_phrases[index], flag)
                #say("Hi! Is anybody here?", flag)
            elif ok==True:
                say("I cannot detect your eyes, could you please open your eyes?", flag)
            while flag.isSet():
                _, frame,tracker, ok = detectFace(frame,tracker,face_cascade,eye_cascade)
                probStream = None
                rval, frame = vc.read()
                cv2.imshow("Webcam Preview", frame)
                cv2.waitKey(1)
                
            #if process == None:
            #    process = subprocess.Popen(['rhythmbox', 'creepyMusic.mp3'])
        else:
            saidNothing += 1
            
        # Re-Initialise Tracker after predetermined Tracking period 'Tracking_Period' (seconds)
        t2 = cv2.getTickCount()
        if (t2 - t1)/ cv2.getTickFrequency()>Tracking_Period:
            ok,t1,tracker=Re_Init_Tracker(frame,face_cascade)

        ################################################## 
        # Tracker Re-initialisation complete
        ##################################################
        cv2.imshow("Webcam Preview", frame)
        cv2.waitKey(1)
        rval, frame = vc.read()
    roboFace.neutral()
    cv2.destroyWindow("Webcam Preview")

    # Black_Hair Blond_Hair Brown_Hair Eyeglasses Gray_Hair Male
    # Mouth_Slightly_Open No_Beard Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Lipstick
