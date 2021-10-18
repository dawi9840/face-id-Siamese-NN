import os
os.environ['TF_cpp_MIN_LEVEL'] =  '2'

import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

# Import uuid library to generate unique image names
import uuid


def negative_dataset():
    # Move LFW Images to the following repository data/negative
    for directory in os.listdir('lfw'):
        for file in os.listdir(os.path.join('lfw', directory)):
            EX_PATH = os.path.join('lfw', directory, file)
            NEW_PATH = os.path.join(NEG_PATH, file)
            os.replace(EX_PATH, NEW_PATH)


def camera_save_img_to_POS_and_ANC_PATH():
    # Establish a connection to the webcam
    cap = cv2.VideoCapture(0)
    while cap.isOpened(): 
        ret, frame = cap.read()

        # Cut down frame to 250x250px
        frame = frame[120:120+250, 200:200+250, :]
        
        # Collect anchors 
        if cv2.waitKey(1) & 0XFF == ord('a'):
            print('take a image to anchors.')
            # Create the unique file path 
            imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)
            
        # Collect positives
        if cv2.waitKey(1) & 0XFF == ord('p'):
            print('take a image to positives.')
            # Create the unique file path 
            imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
            # Write out positive image
            cv2.imwrite(imgname, frame)

        # Show image back to screen
        cv2.imshow('Image Collection', frame)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    # Setup paths
    POS_PATH = os.path.join('data', 'positive')
    NEG_PATH = os.path.join('data', 'negative')
    ANC_PATH = os.path.join('data', 'anchor')

    # # Make the directories
    # os.makedirs(POS_PATH)
    # os.makedirs(NEG_PATH)
    # os.makedirs(ANC_PATH)

    # negative_dataset()
    camera_save_img_to_POS_and_ANC_PATH()




