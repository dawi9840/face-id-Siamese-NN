# face-id-Siamese-NN
Face recognitionwith a Siamese neural network.  

Need create a data folder include anchor, negative, positive folders, and unzip the file which move images to negative folder from download **Labeled Faces in the Wild Home dataset** [All images as gzipped tar file](http://vis-www.cs.umass.edu/lfw/#download).  
And use **face_data_create.py** to create and collect face image dataset to anchor and positive folder.  


* The folder structure:
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
    │
    ├── model_weights
    │
    └── training_checkpoints
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
