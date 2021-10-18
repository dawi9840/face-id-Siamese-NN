# face-id-Siamese-NN
Face recognitionwith a Siamese neural network.  

Need create a data folder which include anchor, negative, positive folders, and unzip the lfw file move images to negative folder from download [Labeled Faces in the Wild Home](vis-www.cs.umass.edu/lfw/lfw.tgz).  
And using **face_data_create.py** to create and collect face image dataset to anchor and positive floder

* folder structure:
```bash
    │
    ├── data
    │   └── coco_2017_dataset
    │       ├── anchor/*
    │       ├── negative/*
    │       └── positive/*
    │
    └── application_data
    │       ├── input_image/*   
    │       └──verification_images/*
```

Paper:[Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)  

## Install  

**Conda virtual env**  
```bash

conda create --name [env_name]  python=3.8
conda activate [env_name]
pip install tensorflow-gpu==2.6.0
conda install cudnn==8.2.0.53
pip install mediapipe==0.8.6.2
pip install matplotlib
pip install opencv-python
```
