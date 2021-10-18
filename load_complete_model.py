import os
import sys
os.environ['TF_cpp_MIN_LEVEL'] =  '2'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

def gpu_setting():
    # Avoid OOM errors by setting GPU Memory Consumption Growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPU: {gpus}')
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True)


def img_preprocessing(file_path):
    '''Scale and Resize'''
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))

    # Scale image to be between 0 and 1 
    img = img / 255.0
    return img


def split_twin_img_set(input_img, validation_img, label):
    '''Build Train and Test Partition'''
    return (img_preprocessing(input_img), img_preprocessing(validation_img), label)


def data_preprocessing(anchor, positive, negative):
    '''data preprocessing'''
    #  Create Labelled Dataset
    # positives => anchor, positive, 1, ..., anchor, positive, 1    # (anchor, positive, 1) => 1,1,1,1,1 #
    # negatives => anchor, negative, 0, ..., anchor, negative, 0    # (anchor, negative, 0) => 0,0,0,0,0 #
    # data => anchor, positive, 1, ..., anchor, positive, 1, anchor, negative, 0, ..., anchor, negative, 0

    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)

    # samples = data.as_numpy_iterator()
    # exampple = samples.next()
    # # tf.print(f'exampple: {exampple}')

    # res = split_twin_img_set(*exampple)
    # # plt.imshow(res[1])
    # # plt.show()

    # Build dataloader pipeline
    data = data.map(split_twin_img_set)
    data = data.cache()
    data = data.shuffle(buffer_size=1024)

    # Training partition
    train_data = data.take(round(len(data)*.7))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    # Testing partition
    test_data = data.skip(round(len(data)*.7))
    test_data = test_data.take(round(len(data)*.3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)

    return train_data, test_data


def visualization_result(test_input, test_val):
    '''Visualization the result.(positive or negative test)
    where i is the ranage 1~ 16.
    Input:
    visualization_result(test_input=test_input[i], test_val=test_val[i])
    
    Output:
    Visualization show the positive or negative result.
    '''
    # Set plot size 
    plt.figure(figsize=(18,8))

    # Set first subplot
    plt.subplot(1, 2, 1) # subplot(nrows, ncols, index, **kwargs)
    plt.imshow(test_input)

    # Set second subplot
    plt.subplot(1, 2, 2)
    plt.imshow(test_val)

    # Renders cleanly
    plt.show()


def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        # print(f'img: {image}')
        input_img = img_preprocessing(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = img_preprocessing(os.path.join('application_data', 'verification_images', image))
        # print(f'validation_img_path: {validation_imge}\n')
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
    verified = verification > verification_threshold
    
    return results, verified


def camera_verification(cap):
    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'w: {w}, h: {h}')

    while cap.isOpened():
            ret, frame = cap.read()
            frame = frame[120:120+250,200:200+250, :]
            cv2.imshow('Verification', frame)

            # Verification trigger
            if cv2.waitKey(10) & 0xFF == ord('v'):
                # Save input image to application_data/input_image folder 
                cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
                # Run verification
                results, verified = verify(model, 0.9, 0.7)
                # print(f'results: {results}')
                print(f'verified: {verified}')

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    print('Done.')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    gpu_setting()

    # Setup paths
    POS_PATH = os.path.join('data', 'positive')
    NEG_PATH = os.path.join('data', 'negative')
    ANC_PATH = os.path.join('data', 'anchor')

    anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(300)
    positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(300)
    negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(300)

    _, test_data = data_preprocessing(anchor, positive, negative)

    test_input, test_val, y_true = test_data.as_numpy_iterator().next()

    model = tf.keras.models.load_model('model_weights/complete_model/10.15')
    # print(model.summary())

    y_hat = model.predict([test_input, test_val])
    # print(f'prediction_prob:\n{y_hat}')

    # Method1: post processing the result
    result = [1 if prediction > 0.5 else 0 for prediction in y_hat]
    # print(f'prediction: {result}')

    # visualization_result(test_input=test_input[0], test_val=test_val[0])

    camera_verification(cap=cv2.VideoCapture(0))


    
