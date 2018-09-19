Roboface is a Humanoid social Robot created by students at the Robotics Lab at Uni-Heidelberg
in order to study robotics and Human-robot interaction.
Briefly the robot recieves an image by a videocamera on its forehead. It detects and tracks 
the face of the closest human and generates behaviours by a pretrained convolutional neural network.
The neural network can distinguish between classes like males,females, long, short hair etc.
It reacts by saying complemets to people. It speaks with espeak TTS sythesizer and soft silicon lips.
Durring summer semester 2018 the code was optimised to run on a raspberry pi 3 B+ computer.
A face tracker based on openCV MEdianfow was added and now the robot only detects tracks and
speaks to the closest human. Lastly, we imaplemeted a Graphical User Interface (GUI) with TKinter library
in order to give interactive Demos with a Thouchscreen. Also A parser has been implemented to create an easier
way to creat quick custom demos for the robot.

#COMMANDS to start

# for the Interactive Gui version 
cd Desktop/RoboFacePi/face_detection/
python2 run_pi_gui.py 

(or python2 run_pi.py for the non gui version optimised for raspberry pi 3 B+)

#for the parser Demo 
cd Desktop/RoboFacePi/Demos2018/
python2 New_Demo.py 



