import os
import sys
os.environ['TF_cpp_MIN_LEVEL'] =  '2'

# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision, Recall
import numpy as np

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


def make_embedding_model(): 
    inp = Input(shape=(100,100,3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')


class L1Dist(Layer):
    '''Build Distance Layer
    # Siamese L1 Distance class'''
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def make_siamese_model(embedding_model): 
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding_model(input_image), embedding_model(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


@tf.function
def train_step(batch, model):
    '''Build Train Step Function
    # Record all of our operations'''
    with tf.GradientTape() as tape:     
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # Forward pass
        # yhat = siamese_model(X, training=True)
        yhat = model(X, training=True)

        # Calculate loss
        loss = tf.losses.BinaryCrossentropy()(y, yhat)
    print(f'loss: {loss}')
        
    # Calculate gradients
    grad = tape.gradient(loss, model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, model.trainable_variables))
    
    return loss


def train(data, EPOCHS, model):
    '''Build Training Loop'''
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            # print(f'type data: {type(data)}')
            # print(f'batch: {type(batch)}')
            train_step(batch, model)
            progbar.update(idx+1)
        
        # Save checkpoints
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)


def recall_result(y_true, y_hat):
    # Create a materic object
    m = Recall()
    # calculate Recall value
    m.update_state(y_true, y_hat)
    print('Recall result:', (m.result().numpy())*100, '%')


def precision_result(y_true, y_hat):
    # Create a materic object
    m2 = Precision()
    # calculate Precision value
    m2.update_state(y_true, y_hat)
    print('Precision result:', (m2.result().numpy())*100, '%')


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


if __name__ == '__main__':
    gpu_setting()

    # Setup paths
    POS_PATH = os.path.join('data', 'positive')
    NEG_PATH = os.path.join('data', 'negative')
    ANC_PATH = os.path.join('data', 'anchor')
    
    anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(300)
    positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(300)
    negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(300)

    # dir_test = anchor.as_numpy_iterator()
    # next_test = dir_test.next()
    # print(f'------\ndir_test: {dir_test}\nfile_path: {next_test}')
    # img = img_preprocessing('data/positive/5dc14632-24e7-11ec-9c58-1933997e258b.jpg')
    # print(f'img: {img}')
    # img.numpy().max()
    # plt.imshow(img)
    # plt.show()

    '''data preprocessing'''
    #  Create Labelled Dataset
    # positives => anchor, positive, 1, ..., anchor, positive, 1    # (anchor, positive, 1) => 1,1,1,1,1 #
    # negatives => anchor, negative, 0, ..., anchor, negative, 0    # (anchor, negative, 0) => 0,0,0,0,0 #
    # data => anchor, positive, 1, ..., anchor, positive, 1, anchor, negative, 0, ..., anchor, negative, 0

    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)

    # data_exampple = data.as_numpy_iterator().next()
    # # tf.print(f'exampple_data: {data_exampple}')
    # data_input,  data_val, label = split_twin_img_set(*data_exampple)
    # # visualization_result(data_input, data_val)
    # # print(label)

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

    '''Model Build'''
    model = make_embedding_model()
    # print(model.summary())

    # Make Siamese Model
    siamese_model = make_siamese_model(embedding_model=model)
    # print(siamese_model.summary())

    '''Model training'''
    EPOCHS = 50
    opt = tf.keras.optimizers.Adam(1e-4) # 0.0001
    
    # Establish Checkpoints
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

    # model training
    train(data=train_data, EPOCHS=EPOCHS, model=siamese_model)

    '''Model evaluate''' 
    test_input, test_val, y_true = test_data.as_numpy_iterator().next()
    # print(test_data.as_numpy_iterator().next())

    # # Get a batch of test data.
    # test_var = test_data.as_numpy_iterator().next()
    # print(len(test_var[0]))  # 16 input
    # print(test_var[2])       # label

    # Make a predictions
    y_hat = siamese_model.predict([test_input, test_val])
    print(f'prediction_prob:\n{y_hat}')

    # Method1: post processing the result
    print('prediction:',[1 if prediction > 0.5 else 0 for prediction in y_hat])

    # # Method2: post processing the result
    # result = []
    # for prediction in y_hat:
    #     if prediction > 0.5:
    #         result.append(1)
    #     else:
    #         result.append(0)
    # print(f'prediction: {result}')

    # Materic: Recall and Precision.
    recall_result(y_true=y_true, y_hat=y_hat)
    precision_result(y_true=y_true, y_hat=y_hat)

    # visualization_result(test_input=test_input[0], test_val=test_val[0])

    '''Model save '''
    siamese_model.save('./model_weights/siamese_model_10.14.h5') # save weights
    print('Save HDF5 model done!')