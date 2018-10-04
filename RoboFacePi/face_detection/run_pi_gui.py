
# coding: utf-8

# In[1]:

import sys
if sys.version_info[0] == 2:
    import Tkinter as tk
    import tkFont as font
else:
    import tkinter as tk
    from tkinter import font
import PIL.Image, PIL.ImageTk

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


# In[2]:

class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened(): raise ValueError("Unable to open video source", video_source)        
        self.width, self.height = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) # Get video source width and height
        self.ind_run_demo, self.ind_recog_face, self.ind_track_face, self.ind_caption = 0, 0, 0, 0
        self.caption = ""        
        
        self.roboFace = face.Face(x_weight=0.8, y_weight=0.2)
        #################################################################
        # Set up tracker
        self.tracker = cv2.TrackerMedianFlow_create()
        self.Tracking_Period=5 # set tracking period before re-initialisation in seconds
        # Load Neural Net model and meanFace 
        self.model = load_model('../face_detection/trained/pretrained_CelebA_normalised0203-05.h5')
        self.meanFace = np.load('../face_detection/mean_face_normalised.npy')
        # Load Face Cascade and Eye Cascade classifiers
        self.face_cascade = cv2.CascadeClassifier('../face_detection/haarcascade_frontalface_alt.xml')
        self.eye_cascade = cv2.CascadeClassifier('../face_detection/haarcascade_eye.xml')
        #################################################################
        # Set Speed for smoother movement
        self.roboFace.setSpeedAll(100)
        self.roboFace.setSpeedHead(80)
        self.flag = Event()
        self.flag.clear()
        #################################################################
        self.roboFace.neutral()
        self.probStream = None
        self.saidNothing = 0
        self.t1 = cv2.getTickCount()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.waiting_phrases=["Hi! Is anybody here?","Greetings human! Nice to meet you! ","My name is roboface! I am a friendly robot!","Hello! It's a pleasure to meet you!","I feel so lonely!"]
        
    def __del__(self): # Release the video source when the object is destroyed
        if self.vid.isOpened(): self.vid.release()
        
    def get_frame(self):        
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if self.ind_run_demo == 1:  
                self.ind_track_face = 0 # turn off independent tracker if demo is on
                if self.flag.isSet() == False:
                    self.caption=""
                    self.run_demo(frame)
            if self.flag.isSet(): _, _, _ = self.detectFace(frame);
            if self.ind_recog_face == 1: self.recog_faces(frame);
            if self.ind_track_face == 1: _, _, _ = self.detectFace(frame);
            if self.ind_caption == 1: self.show_caption(frame)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#            frame = resize(frame, (self.canvas_width,600), mode='edge')
            
            # Return a boolean success flag and the current frame converted to BGR
#            if ret: return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if ret: return (ret, frame)
            else: return (ret, None)
        else: return (ret, None)        

    def imgCrop(self, image, cropBox, boxScale=1):
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

    # Normalize faces usinginter-ocular distance i.o.d
    def normaliseImage(self, image, eyes, xcrop, ycrop):
        # resite, such that i.o.d is always same
        left_eye, right_eye = eyes[0] + np.array([xcrop, ycrop, 0, 0]), eyes[1] + np.array([xcrop, ycrop, 0, 0])
        scale = IOD / np.linalg.norm(left_eye - right_eye)
        left_eye, right_eye = scale * left_eye, scale * right_eye
        im = resize(image, (int(scale * image.shape[0]), int(scale * image.shape[1])), mode='edge')

        # rotate to keep inter ocular line horizontal
        diff = np.subtract(left_eye, right_eye)
        angle = math.atan2(diff[0], diff[1])
        im = rotate(im, -angle, center=(left_eye[0], left_eye[1]), preserve_range=True, mode='edge')

        # new resizing for making the image compatible with the trained NN.
        iod = np.linalg.norm(left_eye - right_eye)
        xmin, xmax = int(left_eye[0] - 1.6 * iod), int(left_eye[0] + 2 * iod)
        ymin, ymax = int(left_eye[1] - 1.3 * iod), int(right_eye[1] + 1.3 * iod)
        xmin, xmax = max(0, xmin), min(im.shape[0], xmax)
        ymin, ymax = max(0, ymin), min(im.shape[1], ymax)
        im = im[xmin:xmax, ymin:ymax, :]
        try:
            im = resize(im, IMAGE_SIZE, mode='edge')
        except:
            return None

        return im
    
    #######################################################################
    # Definition of Track_face: 
    # Tracks only the face of the active user with MEDIANFLOW at opencv 
    # and returns the bounding box of the face
    #######################################################################
    def Track_face(self, frame):
        ok, bbox = self.tracker.update(frame)
        if ok==False:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,20), self.font, 0.75,(0,0,255),2)
            ok, bbox, faces = self.init_tracker(frame)
            print("Track_Face - Tracker Re-Initialisation")
            print("Number of Detected faces: ",len(faces))
        return ok, bbox

    def find_faces(self, frame):
        bbox = None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        #calculate the rectangle with the biggest area
        #biggest area means that its the closest face to the robot
        max_area=0
        for face in faces:
            area=face[2]*face[3]
            if area>max_area:
                max_area=area
                bbox=tuple(face)
        return bbox, faces # return biggest box and all faces
    
    # Draw bounding box on all faces
    def recog_faces(self, frame):
        _, faces = self.find_faces(frame)
        for face in faces:
            (x, y, w, h) = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
       
    # initialize tracker
    def init_tracker(self, frame):
        bbox, faces = self.find_faces(frame)
        self.tracker = cv2.TrackerMedianFlow_create()
        # re-initialise the tracker in case of lost target
        ok = self.tracker.init(frame, bbox)
        if len(faces)==0: ok=False            
        return ok, bbox, faces
    
    # Re-initialization of Tracker after given period of time
    def Re_Init_Tracker(self, frame):
        cv2.putText(frame, "Re-Initialising Tracking", (100,50), self.font, 0.75,(255,0,0),2)
        ok, _, faces = self.init_tracker(frame)
        print("Tracker Re-Initialisation")
        print("Number of Detected faces: ",len(faces))
        t1 = cv2.getTickCount()
        return ok, t1

    def detectFace(self, image):
        # http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        unaltered_image = image.copy()
        eyes = None
        normalised_image = None
        # Track face in current frame
        ok, bbox = self.Track_face(image)
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
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) == 2:
                left_eye = eyes[0][0:2] + x
                right_eye = eyes[1][0:2] + y
                eyex = int((left_eye[0] + right_eye[0]) * .5)
                eyey = int((left_eye[1] + right_eye[1]) * .5)
                self.roboFace.moveHead(eyex, eyey)
            # suggestion: skip this frame as prediction, so return None, image
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                if len(eyes) == 2 and np.abs(eyes[0, 1] - eyes[1, 1]) < 10:
                    offset1 = np.sqrt((eyes[0, 2] ** 2 + eyes[0, 3] ** 2)) * 0.5
                    offset2 = np.sqrt((eyes[1, 2] ** 2 + eyes[1, 3] ** 2)) * 0.5
                    real_eyes = eyes + np.array([[x + offset1, y + offset1, 0, 0], [x + offset2, y + offset2, 0, 0]])
                    real_eyes = np.sort(real_eyes, axis=0)
                    cropped_image, xcrop, ycrop = self.imgCrop(unaltered_image, bbox)
                    normalised_image = self.normaliseImage(cropped_image, real_eyes, -xcrop, -ycrop)

        return normalised_image, image, ok

    def mapAttributes(self, classes):
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
    def Undersampled_Lip_Tragectory(self, phrase, Sleep_Time):
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

    def MoveLips(self, Sleep_Time, Amplitude, flag):
        self.roboFace.setSpeedLips(127)
        sleep(0.5)
        i = 0
        while flag.isSet() and i < len(Amplitude):
            self.roboFace.moveLips(int(Amplitude[i]))
            sleep(Sleep_Time)
            i = i + 1

        if ~flag.isSet():
            self.roboFace.moveLips(0)
            sleep(0.05)

    def Talk(self, phrase, flag):
        A = "espeak -z -s 80 -v female5 "
        A = A + "'" + phrase + "'"
        os.system(A)
        flag.clear()

    def say(self, phrase, flag):
        phrase = phrase.replace("'", " ")
        self.caption=phrase
        flag.set()
        Sleep_Time = 0.05
        Amplitude, Time = self.Undersampled_Lip_Tragectory(phrase, Sleep_Time)
        thread_movement = Thread(target=self.MoveLips, args=(Sleep_Time, Amplitude, flag))
        thread_talk = Thread(target=self.Talk, args=(phrase, flag))
        thread_talk.start()
        thread_movement.start()
    ################################################################################
    # End of Declaration: say - Talk - MoveLips 
    ################################################################################

    def sayDoSomething(self, pred_attr):
        talk = {'Smiling': 'I like it when people smile at me!',
                'Female': 'You are a female, am I right?',
                'Male': 'You are a male, am I right?',
                'Wearing_Earrings': 'You are wearing beautiful earrings today!',
                'Wearing_Lipstick': 'I see you are wearing lipstick today. Pretty!',
                'Blond_Hair': 'Nice blond hair!',
                'Eyeglasses': 'You are wearing eyeglasses!',
                'Brown_Hair': 'You have nice brown hair!',
                'Black_Hair': 'You have nice black hair!',
                'Gray_Hair': 'Gray Hair! You must be a wise man!',
                'Wavy_Hair': 'You have nice wavy hair!',
                'Straight_Hair': 'You have nice straight hair.'
                }

        if 'Smiling' in pred_attr:
            self.roboFace.happy(moveHead=False,movelips=False)
        elif 'Black_Hair' in pred_attr:
            self.roboFace.angry(moveHead=False,movelips=False)
        elif 'Eyeglasses' in pred_attr:
            self.roboFace.unsure(moveHead=False,movelips=False)
        else:
            self.roboFace.neutral(moveHead=False,movelips=False)

        index = np.random.randint(0, len(pred_attr))
        self.say(talk[pred_attr[index]], self.flag)

    def getProbaStream(self, probStream, probs):
        if probStream == None:
            probStream = probs
        else:
            probStream = np.vstack((probStream, probs))
        return probStream
    
    def run_demo(self, frame):
        normalised_image, frame, ok = self.detectFace(frame)
        # if a face is detected and the normalisation was successful, predict on it
        if normalised_image is not None:
            normalised_image = normalised_image[:, :, ::-1]
            # subtract mean face
            X_test = np.expand_dims(normalised_image, axis=0)
            X_test -= self.meanFace
            classes = self.model.predict_classes(X_test, batch_size=32, verbose=0)
            proba = self.model.predict_proba(X_test, batch_size=32, verbose=0)
            # pred_attr = mapAttributes((proba > 0.6)[0])
            # print( proba)
            # print(pred_attr)

            self.probStream = self.getProbaStream(self.probStream, proba)
            if self.saidNothing == 0 and self.probStream.shape[0] < 10:
                self.saidNothing += 1
                ret, frame = self.vid.read()

            elif self.probStream.shape[0] > 10 and len(self.probStream.shape) >= 2:
                meanProbs = np.mean(self.probStream, axis=0)
                pred_attr = self.mapAttributes(meanProbs > 0.6)
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
                if meanProbs[5] < 0.25:
                    best.append('Female')
                elif meanProbs[12] < 0.11 and meanProbs[11] < 0.11 and meanProbs[5] > 0.85:
                    best.append('Male')
                print(meanProbs)
                print("BEST", best)

                # end NN stuff
                # postprocessing and reaction step
                self.sayDoSomething(best)
                self.saidNothing = 0
                #while self.flag.isSet():
                #    _, frame, ok = self.detectFace(frame)
                #    self.probStream = None
                #    ret, frame = self.vid.read()

        elif self.saidNothing > 150:
            self.saidNothing = 0
            self.roboFace.sad()
            if ok==False:
                index = np.random.randint(0, len(self.waiting_phrases))
                self.say(self.waiting_phrases[index], self.flag)
                #say("Hi! Is anybody here?", flag)
            elif ok==True:
#                self.say("I cannot detect your eyes.", self.flag)
                self.say("Could you please open your eyes?", self.flag)
           # while self.flag.isSet():
           #     _, frame, ok = self.detectFace(frame)
           #     self.probStream = None
           #     ret, frame = self.vid.read()
                

            #if process == None:
            #    process = subprocess.Popen(['rhythmbox', 'creepyMusic.mp3'])
        else:
            self.saidNothing += 1

        # Re-Initialise Tracker after predetermined Tracking period 'Tracking_Period' (seconds)
        t2 = cv2.getTickCount()
        if (t2 - self.t1) / cv2.getTickFrequency() > self.Tracking_Period:
            ok, self.t1 = self.Re_Init_Tracker(frame)
            
    def recog_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for face in faces:
            (x, y, w, h) = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def show_caption(self, frame):        
        textlen = cv2.getTextSize(self.caption, self.font, 1, 2)[0][0]
        posx, posy = int((self.width-textlen)/2), int(self.height-20)
        cv2.putText(frame, self.caption, (posx,posy), self.font, 0.75,(0,0,255),2)


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
                
#        self.vid = MyVideoCapture('testvideo.avi') # open video source
        self.vid = MyVideoCapture()
        # Create a canvas that can fit the above video source size
        self.canvas_width, self.canvas_height = 1024, 552 # reduced by the 2 pixels of menu bar
#        self.canvas_width, self.canvas_height = 1024, 640
        self.canvas = tk.Canvas(self.window, width = self.canvas_width, height = self.canvas_height)
#        self.canvas = tk.Canvas(self.window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
       
        # values to accommodate for the title label
        self.lblxlenadd, self.lblylen = 10, 31 # 21 corresponds to height=1 in show_title_label
        
        # standard button configuration
        self.btnwidth, self.btnheight = 18, 3
        self.btnxlen, self.btnylen = 192, 60
        self.btnypos_list = [] # menu button y positions
        for i in range(7):
            self.btnypos_list.append(self.lblylen+self.btnylen*i)
            
        self.fonttype = font.Font(family="Helvetica", size=12, weight=font.BOLD)

        # Initialize ALL menu items here
        self.submenu_btnlist = None
        self.btn_demo_r, self.btn_demo_g = None, None # red and green buttons
        self.btn_recog_r, self.btn_recog_g = None, None
        self.btn_tracker_r, self.btn_tracker_g = None, None
        self.btn_face = None # make a roboface has a submenu, do not need red or green buttons
        self.btn_face_neutral, self.btn_face_happy, self.btn_face_unsure, self.btn_face_sad = None, None, None, None
        self.btn_face_angry, self.btn_face_angry2 = None, None
        self.btn_caption_r, self.btn_caption_g = None, None
        self.btn_mode_r, self.btn_mode_g, self.btn_mode_value = None, None, False
        self.btn_mode_pin, self.btn_mode_user = 1234, 0 # PIN number to toggle to fullscreen mode
        self.btn_mode_1, self.btn_mode_2, self.btn_mode_3 = None, None, None
        self.btn_mode_4, self.btn_mode_5, self.btn_mode_6, self.btn_mode_enter = None, None, None, None
        self.btn_quit = None
        self.btn_quit_pin, self.btn_quit_user = 1234, 0 # PIN number to quit
        self.btn_quit_1, self.btn_quit_2, self.btn_quit_3 = None, None, None
        self.btn_quit_4, self.btn_quit_5, self.btn_quit_6, self.btn_quit_enter = None, None, None, None
        self.btn_back = None
        
        # List main menu items here AFTER individual submenu items are declared above
        #              red button object  green button object   button text     method activated by btn
        self.mmenu = [[self.btn_demo_r,   self.btn_demo_g,   "Demo",            self.set_demo_value],
                      [self.btn_recog_r,  self.btn_recog_g,  "Face Detection",  self.set_recog_value],
                      [self.btn_tracker_r,self.btn_tracker_g,"Face Tracker",    self.set_track_value],
                      [self.btn_face,     None,              "Make a RoboFace", self.roboface_submenu],
                      [self.btn_caption_r,self.btn_caption_g,"Caption",         self.set_caption_value],
                      [self.btn_mode_r,   self.btn_mode_g,   "Fullscreen Mode", self.mode_submenu],
                      [self.btn_quit,     None,              "Quit",            self.quit_submenu]]
        self.mainmenu()
        self.mmenu_btn_r, self.mmenu_btn_g, self.mmenu_txt, self.mmenu_cmd = self.build_list(self.mmenu)
        
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1
        self.update()
        self.window.mainloop()
        
    def build_list(self, menulist):
        btnlist_r, btnlist_g, txtlist, cmdlist = [], [], [], []
        for i in menulist:
            btnlist_r.append(i[0])
            btnlist_g.append(i[1])
            txtlist.append(i[2])
            cmdlist.append(i[3])
        return btnlist_r, btnlist_g, txtlist, cmdlist

    # This is the looping method that updates the canvas with image obtained from the
    #  video class via self.vid.get_frame
    def update(self):
        ret, frame = self.vid.get_frame()
        if ret: 
            self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        self.window.after(self.delay, self.update)

    # A title label is added at the top of the button list to distinguish btw main- and sub-menu
    def show_title_label(self, title):
        self.lbl_title = tk.Label(text=title, width=25, height=2, bg="yellow", anchor="w", font=self.fonttype)
        self.lbl_title.place(x=self.canvas_width-self.btnxlen, y=0)
        
    # "Back" button is only added when there is a submenu
    # it is added at the end of all submenu buttons, and is smaller, to distinguish it from others
    def add_back_button(self,submenu_btnlist):
        if self.btn_back is None:
            self.btn_back = tk.Button(self.window, text="Back", width=16, height=2,
                                           font=self.fonttype, command=self.back_roboface_submenu)
        self.btn_back.place(x=self.canvas_width-170, y=self.lblylen+self.btnylen*len(submenu_btnlist)+5)
        self.submenu_btnlist = submenu_btnlist

    # When toggling between main and submenu, we use this method to hide the other buttons
    def hide_buttons(self,btnlist):
        for i in btnlist:
            if i is not None:
                i.place_forget()

    # depending on the val parameter, we set the color and whether we want the red/green button
    # num1 or num2 = 0 means we want the red button (corresponding to the self.mmenu element)
    # num1 or num2 = 1 means green button
    # num1 is the button we want to show, num2 is the button we want to hide
    def button_val(self, val):
        if val == -1:
            num1 = 0; num2 = 0; color = "gray";
        elif val == 1:
            num1 = 1; num2 = 0; color = "green";
        elif val == 0:
            num1 = 0; num2 = 1; color = "red";
        return num1, num2, color
    
    # create a button
    def button(self, menu_list, i, val, txt=None):
        num1, num2, color = self.button_val(val)
        if txt is None: txt = menu_list[i][2]
        if val == 1: txt = txt + "\nis on";
        elif val == 0: txt = txt + "\nis off";
        if menu_list[i][num1] is None:
            menu_list[i][num1] = tk.Button(self.window, width=self.btnwidth, height=self.btnheight, #wraplength=150, 
                                                fg="white", bg=color, activebackground=color, font=self.fonttype,
                                                text=txt, command=menu_list[i][3])
        menu_list[i][num1].place(x=self.canvas_width-self.btnxlen, y=self.btnypos_list[i])
        if color == "green" or color == "red":
            menu_list[i][num2].place_forget()

    # create all buttons in the menu list
    def menu_buttons(self, menu_list, submenu):
        for i in range(len(menu_list)):
            val = -1
            if submenu != 1:
                if i == 0: val = self.vid.ind_run_demo;
                elif i == 1: val = self.vid.ind_recog_face;
                elif i == 2: val = self.vid.ind_track_face;
                elif i == 4: val = self.vid.ind_caption;
                elif i == 5: val = self.btn_mode_value;
            num, _, color = self.button_val(val)
            if submenu == 1: color = "gray";
            txt = menu_list[i][2]
            if val == 1: txt = txt + "\nis on";
            elif val == 0: txt = txt + "\nis off";
            if menu_list[i][num] is None:
                menu_list[i][num] = tk.Button(self.window, width=self.btnwidth, height=self.btnheight, #wraplength=80,
                                                fg="white", bg=color, activebackground=color, font=self.fonttype,
                                                text=txt, command=menu_list[i][3])
            menu_list[i][num].place(x=self.canvas_width-self.btnxlen, y=self.btnypos_list[i])
        
    # create main menu buttons
    def mainmenu(self):
        self.show_title_label("Main Menu")
        if self.submenu_btnlist is not None:
            self.btn_back.place_forget()
            self.hide_buttons(self.submenu_btnlist)
            self.submenu_btnlist = None        
        self.menu_buttons(self.mmenu,0)

    # create "Make a Roboface" submenu
    def submenu(self, title_label, menu_list):
        self.show_title_label(title_label)
        self.hide_buttons(self.mmenu_btn_r)
        self.hide_buttons(self.mmenu_btn_g)
        self.menu_buttons(menu_list,1)
        btn_list, _, _, _ = self.build_list(menu_list)
        self.add_back_button(btn_list)        
       
    def set_demo_value(self): # method tied to the "Demo" button
        self.vid.ind_run_demo = not self.vid.ind_run_demo
        if self.vid.ind_run_demo == 1:
            if self.vid.ind_track_face == 1: self.set_track_value() # if tracker running, turn it off
        elif self.vid.ind_run_demo == 0:
            self.vid.roboFace.neutral()
        self.button(self.mmenu, 0, self.vid.ind_run_demo)
        
    def set_recog_value(self): # method tied to the "Face Detection" button
        self.vid.ind_recog_face = not self.vid.ind_recog_face
        self.button(self.mmenu, 1, self.vid.ind_recog_face)

    def set_track_value(self): # method tied to the "Face Tracker" button
        self.vid.ind_track_face = not self.vid.ind_track_face
        if self.vid.ind_track_face == 1:
            if self.vid.ind_run_demo == 1: self.set_demo_value() # if demo running, turn it off
        self.button(self.mmenu, 2, self.vid.ind_track_face)
            
    def roboface_submenu(self): # method tied to the "Make a RoboFace" button
        if self.vid.ind_run_demo == 1:   self.set_demo_value() # if demo is on, turn it off
        if self.vid.ind_recog_face == 1: self.set_recog_value() # if recog is on, turn it off
        if self.vid.ind_track_face == 1: self.set_track_value() # if tracker running, turn it off
        if self.vid.ind_caption == 1:    self.set_caption_value() # if caption is on, turn it off
        self.submenu("Make a RoboFace",
                     [[self.btn_face_neutral, None, "Neutral", self.vid.roboFace.neutral],
                      [self.btn_face_happy,   None, "Happy",   self.vid.roboFace.happy],
                      [self.btn_face_unsure,  None, "Unsure",  self.vid.roboFace.unsure],
                      [self.btn_face_sad,     None, "Sad",     self.vid.roboFace.sad],
                      [self.btn_face_angry2,  None, "Angry (No Lips)", self.angry_nolips],
                      [self.btn_face_angry,   None, "Angry",   self.vid.roboFace.angry]])

    def angry_nolips(self):
        self.vid.roboFace.angry(movelips = False)
        
    def back_roboface_submenu(self): # method tied to the "Back" button in RoboFace submenu
        self.vid.roboFace.neutral()
        self.mainmenu()
        
    def set_caption_value(self): # method tied to the "Caption" button
        self.vid.ind_caption = not self.vid.ind_caption
        self.button(self.mmenu, 4, self.vid.ind_caption)

    def mode_submenu(self):
        self.submenu("Fullscreen PIN",
                     [[self.btn_mode_1,     None, "1",     self.set_mode_1],
                      [self.btn_mode_2,     None, "2",     self.set_mode_2],
                      [self.btn_mode_3,     None, "3",     self.set_mode_3],
                      [self.btn_mode_4,     None, "4",     self.set_mode_4],
                      [self.btn_mode_5,     None, "5",     self.set_mode_5],
                      [self.btn_mode_6,     None, "6",     self.set_mode_6],
                      [self.btn_mode_enter, None, "Enter", self.set_mode_enter]])
    def set_mode_1(self): self.btn_mode_user += 1000 
    def set_mode_2(self): self.btn_mode_user += 100 
    def set_mode_3(self): self.btn_mode_user += 10 
    def set_mode_4(self): self.btn_mode_user += 1 
    def set_mode_5(self): self.btn_mode_user += 10000 
    def set_mode_6(self): self.btn_mode_user += 10000
    def set_mode_enter(self): 
        if self.btn_mode_pin == self.btn_mode_user:
            self.set_mode_value()
        self.btn_mode_user = 0 # reset user-entered pin
        self.mainmenu()
        
    def set_mode_value(self): # method tied to the "Fullscreen Mode' button
        self.btn_mode_value = not self.btn_mode_value
        self.window.attributes("-fullscreen", self.btn_mode_value)
        self.button(self.mmenu, 5, self.btn_mode_value)

    def quit_command(self): # method tied to "Quit" button
        self.vid.roboFace.neutral()
        self.window.destroy()

    def quit_submenu(self):
        self.submenu("Quit PIN",
                     [[self.btn_quit_1,     None, "1",     self.set_quit_1],
                      [self.btn_quit_2,     None, "2",     self.set_quit_2],
                      [self.btn_quit_3,     None, "3",     self.set_quit_3],
                      [self.btn_quit_4,     None, "4",     self.set_quit_4],
                      [self.btn_quit_5,     None, "5",     self.set_quit_5],
                      [self.btn_quit_6,     None, "6",     self.set_quit_6],
                      [self.btn_quit_enter, None, "Enter", self.set_quit_enter]])
    def set_quit_1(self): self.btn_quit_user += 1000 
    def set_quit_2(self): self.btn_quit_user += 100 
    def set_quit_3(self): self.btn_quit_user += 10 
    def set_quit_4(self): self.btn_quit_user += 1 
    def set_quit_5(self): self.btn_quit_user += 10000 
    def set_quit_6(self): self.btn_quit_user += 10000
    def set_quit_enter(self): 
        if self.btn_quit_pin == self.btn_quit_user:
            self.quit_command()
        else:
            self.btn_quit_user = 0 # reset user-entered pin
            self.mainmenu()
        
App(tk.Tk(), "Webcam") # Create a window and pass it to the Application object
