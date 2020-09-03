# **CS 6301.OU1 - Project 2b**

Name: Alexander Neal, netID: ajn170130


*This notebook uses TensorFlow / Keras to perform CNN image classification on the HiRise Orbital Data Set, which consists of images of the Mars surface.*

## **Import Libraries**


```
# TensorFlow and tf.keras
from tensorflow import keras
from tensorflow.keras import layers
!pip install -U keras-tuner
from kerastuner.tuners import RandomSearch


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
from PIL import Image
from tqdm import tqdm
```

    Collecting keras-tuner
    [?25l  Downloading https://files.pythonhosted.org/packages/a7/f7/4b41b6832abf4c9bef71a664dc563adb25afc5812831667c6db572b1a261/keras-tuner-1.0.1.tar.gz (54kB)
    [K     |████████████████████████████████| 61kB 4.3MB/s 
    [?25hRequirement already satisfied, skipping upgrade: future in /usr/local/lib/python3.6/dist-packages (from keras-tuner) (0.16.0)
    Requirement already satisfied, skipping upgrade: numpy in /usr/local/lib/python3.6/dist-packages (from keras-tuner) (1.18.5)
    Requirement already satisfied, skipping upgrade: tabulate in /usr/local/lib/python3.6/dist-packages (from keras-tuner) (0.8.7)
    Collecting terminaltables
      Downloading https://files.pythonhosted.org/packages/9b/c4/4a21174f32f8a7e1104798c445dacdc1d4df86f2f26722767034e4de4bff/terminaltables-3.1.0.tar.gz
    Collecting colorama
      Downloading https://files.pythonhosted.org/packages/c9/dc/45cdef1b4d119eb96316b3117e6d5708a08029992b2fee2c143c7a0a5cc5/colorama-0.4.3-py2.py3-none-any.whl
    Requirement already satisfied, skipping upgrade: tqdm in /usr/local/lib/python3.6/dist-packages (from keras-tuner) (4.41.1)
    Requirement already satisfied, skipping upgrade: requests in /usr/local/lib/python3.6/dist-packages (from keras-tuner) (2.23.0)
    Requirement already satisfied, skipping upgrade: scipy in /usr/local/lib/python3.6/dist-packages (from keras-tuner) (1.4.1)
    Requirement already satisfied, skipping upgrade: scikit-learn in /usr/local/lib/python3.6/dist-packages (from keras-tuner) (0.22.2.post1)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->keras-tuner) (2020.6.20)
    Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->keras-tuner) (1.24.3)
    Requirement already satisfied, skipping upgrade: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->keras-tuner) (2.9)
    Requirement already satisfied, skipping upgrade: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->keras-tuner) (3.0.4)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->keras-tuner) (0.15.1)
    Building wheels for collected packages: keras-tuner, terminaltables
      Building wheel for keras-tuner (setup.py) ... [?25l[?25hdone
      Created wheel for keras-tuner: filename=keras_tuner-1.0.1-cp36-none-any.whl size=73200 sha256=d6447c07c613ea7a92537e09fcdc1435fc0d25b0579f8fc9ff646f6a5617f533
      Stored in directory: /root/.cache/pip/wheels/b9/cc/62/52716b70dd90f3db12519233c3a93a5360bc672da1a10ded43
      Building wheel for terminaltables (setup.py) ... [?25l[?25hdone
      Created wheel for terminaltables: filename=terminaltables-3.1.0-cp36-none-any.whl size=15356 sha256=5253018f192501bdcb59e7d5d8226a23d44992b9391b41296c571d8994cd5f49
      Stored in directory: /root/.cache/pip/wheels/30/6b/50/6c75775b681fb36cdfac7f19799888ef9d8813aff9e379663e
    Successfully built keras-tuner terminaltables
    Installing collected packages: terminaltables, colorama, keras-tuner
    Successfully installed colorama-0.4.3 keras-tuner-1.0.1 terminaltables-3.1.0




---



## **Data Import and Preprocessing**


Import data and labels into numpy arrays


```
train_url = 'https://ajndata.s3.us-east-2.amazonaws.com/hirise/train.txt'
test_url = 'https://ajndata.s3.us-east-2.amazonaws.com/hirise/test.txt'
img_url = 'https://ajndata.s3.us-east-2.amazonaws.com/hirise/map-proj/'

def get_image(url):
  image = Image.open(urlopen(url))
  return np.array(image)

print("Importing training set...")
train_file = urlopen(train_url).read().decode('utf-8').strip().split('\n')
train_images = []
train_labels = []
for line in tqdm(train_file):
  filename, label = line.split(' ')
  train_images.append(get_image(img_url + filename))
  train_labels.append(int(label))

train_images = np.array(train_images)
train_labels = np.array(train_labels)

print()
print("Importing test set...")
test_file = urlopen(test_url).read().decode('utf-8').strip().split('\n')
test_images = []
test_labels = []
for line in tqdm(test_file):
  filename, label = line.split(' ')
  test_images.append(get_image(img_url + filename))
  test_labels.append(int(label))

test_images = np.array(test_images)
test_labels = np.array(test_labels)

```

    Importing training set...


    100%|██████████| 3056/3056 [23:45<00:00,  2.14it/s]


    
    Importing test set...


    100%|██████████| 764/764 [05:52<00:00,  2.17it/s]


Normalize pixel intensity values to be between 0 and 1:


```
print(train_images.min(), train_images.max())

train_images = train_images/255
test_images = test_images/255
```

    0 255


Reshape image data for Conv2D layer input


```
train_images = train_images.reshape((train_images.shape[0],227,227,1))
test_images = test_images.reshape((test_images.shape[0],227,227,1))

print(train_images.shape)
```

    (3056, 227, 227, 1)


Create a dictionary mapping labels to class names:



```
class_map = {0: 'other',
             1: 'crater',
             2: 'dark_dune',
             3: 'streak',
             4: 'bright_dune',
             5: 'impact',
             6: 'edge'}
```



---



## **Hyperparameter Tuning Phase 1**

We will automate hyperparameter tuning using the `kerastuner` library. For this phase we are concerned with network architecture, optimizer choice, and learning rate. We define the following hyperparameter space:

*   Between 2 and 4 convolutional layers (each followed by a max pooling layer)
*   Either 32 or 64 filters in each convolutional layer
*   After each max pooling layer, no dropout or 25\% dropout
*   1 to 4 dense layers after the final max pooling layer
*   Size of each dense layer ranging from 64 to 512 neurons
*   No dropout or 50\% dropout after each dense layer
*   Initial earning rate 0.001 or 0.0001
*   Adam or RMSProp optimizer

The following function builds and compiles a model based on a set of hyperparameters from the above space.



```
def build_model(hp):
    model = keras.Sequential()

    # convolutional / max pooling layers
    for i in range(hp.Int('num_con_layers', 2, 4)):
      model.add(layers.Conv2D(hp.Choice('filters_' + str(i), values=[32,64]),
                              (3,3), 
                              activation='relu',
                              input_shape=train_images.shape[1:]))

      model.add(layers.MaxPooling2D((2,2)))

      if hp.Choice('drop_con_' + str(i), values=[True,False], default=False):
        model.add(layers.Dropout(0.25))

    # feedforward layers
    model.add(layers.Flatten())
    for i in range(hp.Int('num_ff_layers', 1, 4)):
      model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                          min_value=64,
                                          max_value=512,
                                          step=64),
                                          activation='relu'))
      if hp.Choice('drop_ff_' + str(i), values=[True,False], default=False):
        model.add(layers.Dropout(0.5))
          
    # output layer
    model.add(layers.Dense(7, activation='softmax'))

    # learning rate & optimizer
    learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0001])
    
    if hp.Choice('optimizer', values=['adam', 'rmsprop']) == 'adam':
      model.compile(optimizer=keras.optimizers.Adam(learning_rate),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])
    else:
      model.compile(optimizer=keras.optimizers.RMSprop(learning_rate),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])
      
    model.summary()

    return model
```

Initialize a tuner that will perform a random search of our specified hyperparameter space, then view a summary of the search space.


```
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=30,
    executions_per_trial=3,
    overwrite=True)

tuner.search_space_summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 225, 225, 32)      320       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 112, 112, 32)      0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 110, 110, 32)      9248      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 55, 55, 32)        0         
    _________________________________________________________________
    flatten (Flatten)            (None, 96800)             0         
    _________________________________________________________________
    dense (Dense)                (None, 64)                6195264   
    _________________________________________________________________
    dense_1 (Dense)              (None, 7)                 455       
    =================================================================
    Total params: 6,205,287
    Trainable params: 6,205,287
    Non-trainable params: 0
    _________________________________________________________________



<span style="color:#4527A0"><h1 style="font-size:18px">Search space summary</h1></span>



<span style="color:cyan"> |-Default search space size: 10</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">num_con_layers (Int)</h2></span>



<span style="color:cyan"> |-default: None</span>



<span style="color:blue"> |-max_value: 4</span>



<span style="color:cyan"> |-min_value: 2</span>



<span style="color:blue"> |-sampling: None</span>



<span style="color:cyan"> |-step: 1</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">filters_0 (Choice)</h2></span>



<span style="color:cyan"> |-default: 32</span>



<span style="color:blue"> |-ordered: True</span>



<span style="color:cyan"> |-values: [32, 64]</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">drop_con_0 (Choice)</h2></span>



<span style="color:cyan"> |-default: 0</span>



<span style="color:blue"> |-ordered: True</span>



<span style="color:cyan"> |-values: [1, 0]</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">filters_1 (Choice)</h2></span>



<span style="color:cyan"> |-default: 32</span>



<span style="color:blue"> |-ordered: True</span>



<span style="color:cyan"> |-values: [32, 64]</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">drop_con_1 (Choice)</h2></span>



<span style="color:cyan"> |-default: 0</span>



<span style="color:blue"> |-ordered: True</span>



<span style="color:cyan"> |-values: [1, 0]</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">num_ff_layers (Int)</h2></span>



<span style="color:cyan"> |-default: None</span>



<span style="color:blue"> |-max_value: 4</span>



<span style="color:cyan"> |-min_value: 1</span>



<span style="color:blue"> |-sampling: None</span>



<span style="color:cyan"> |-step: 1</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">units_0 (Int)</h2></span>



<span style="color:cyan"> |-default: None</span>



<span style="color:blue"> |-max_value: 512</span>



<span style="color:cyan"> |-min_value: 64</span>



<span style="color:blue"> |-sampling: None</span>



<span style="color:cyan"> |-step: 64</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">drop_ff_0 (Choice)</h2></span>



<span style="color:cyan"> |-default: 0</span>



<span style="color:blue"> |-ordered: True</span>



<span style="color:cyan"> |-values: [1, 0]</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">learning_rate (Choice)</h2></span>



<span style="color:cyan"> |-default: 0.001</span>



<span style="color:blue"> |-ordered: True</span>



<span style="color:cyan"> |-values: [0.001, 0.0001]</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">optimizer (Choice)</h2></span>



<span style="color:cyan"> |-default: adam</span>



<span style="color:blue"> |-ordered: False</span>



<span style="color:cyan"> |-values: ['adam', 'rmsprop']</span>




---


Perform hyperparameter tuning (30 trials, each 10-epoch trial is repeated thrice)


```
tuner.search(train_images, train_labels, 
             validation_data=(test_images, test_labels),
             epochs = 10)
```



---

View details for the model that performed best during the hyperparameter search


```
best_model = tuner.get_best_models(num_models=1)[0]

best_hp = tuner.get_best_hyperparameters()[0]
print('\n\nLearning Rate: ', best_hp.get('learning_rate'))
print('\nOptimizer: ', best_hp.get('optimizer'))
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 225, 225, 64)      640       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 112, 112, 64)      0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 110, 110, 64)      36928     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 55, 55, 64)        0         
    _________________________________________________________________
    dropout (Dropout)            (None, 55, 55, 64)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 53, 53, 64)        36928     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 26, 26, 64)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 26, 26, 64)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 24, 24, 64)        36928     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 12, 12, 64)        0         
    _________________________________________________________________
    flatten (Flatten)            (None, 9216)              0         
    _________________________________________________________________
    dense (Dense)                (None, 256)               2359552   
    _________________________________________________________________
    dense_1 (Dense)              (None, 7)                 1799      
    =================================================================
    Total params: 2,472,775
    Trainable params: 2,472,775
    Non-trainable params: 0
    _________________________________________________________________
    
    
    Learning Rate:  0.001
    
    Optimizer:  adam





---


# **Hyperparameter Tuning Phase 2**

Using the winning configuration from Phase 1, we now test varying convolutional filter sizes and max pool sizes. The following function builds and compiles a candidate model for tuning:


```
def build_model_2(hp):
    filter_sizes = [3,4,5]
    pool_sizes = [2,3]

    model = keras.Sequential()

    # Convolutional / max pooling layers
    model.add(layers.Conv2D(64,
                            hp.Choice('fsize1', values=filter_sizes),
                            activation='relu',
                            input_shape=train_images.shape[1:]))
    model.add(layers.MaxPooling2D(hp.Choice('psize1', values=pool_sizes)))

    model.add(layers.Conv2D(64,
                            hp.Choice('fsize2', values=filter_sizes),
                            activation='relu'))
    model.add(layers.MaxPooling2D(hp.Choice('psize2', values=pool_sizes)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64,
                            hp.Choice('fsize3', values=filter_sizes),
                            activation='relu'))
    model.add(layers.MaxPooling2D(hp.Choice('psize3', values=pool_sizes)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64,
                            hp.Choice('fsize4', values=filter_sizes),
                            activation='relu'))
    model.add(layers.MaxPooling2D(hp.Choice('psize4', values=pool_sizes)))
    
    # fully connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
          
    # output layer
    model.add(layers.Dense(7, activation='softmax'))

    # learning rate & optimizer

    model.compile(optimizer=keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    model.summary()

    return model
```

Compile a new random search tuner and view search space summary


```
tuner_2 = RandomSearch(
    build_model_2,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=3,
    overwrite=True)

tuner_2.search_space_summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 225, 225, 64)      640       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 112, 112, 64)      0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 110, 110, 64)      36928     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 55, 55, 64)        0         
    _________________________________________________________________
    dropout (Dropout)            (None, 55, 55, 64)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 53, 53, 64)        36928     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 26, 26, 64)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 26, 26, 64)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 24, 24, 64)        36928     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 12, 12, 64)        0         
    _________________________________________________________________
    flatten (Flatten)            (None, 9216)              0         
    _________________________________________________________________
    dense (Dense)                (None, 256)               2359552   
    _________________________________________________________________
    dense_1 (Dense)              (None, 7)                 1799      
    =================================================================
    Total params: 2,472,775
    Trainable params: 2,472,775
    Non-trainable params: 0
    _________________________________________________________________



<span style="color:#4527A0"><h1 style="font-size:18px">Search space summary</h1></span>



<span style="color:cyan"> |-Default search space size: 8</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">fsize1 (Choice)</h2></span>



<span style="color:cyan"> |-default: 3</span>



<span style="color:blue"> |-ordered: True</span>



<span style="color:cyan"> |-values: [3, 4, 5]</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">psize1 (Choice)</h2></span>



<span style="color:cyan"> |-default: 2</span>



<span style="color:blue"> |-ordered: True</span>



<span style="color:cyan"> |-values: [2, 3]</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">fsize2 (Choice)</h2></span>



<span style="color:cyan"> |-default: 3</span>



<span style="color:blue"> |-ordered: True</span>



<span style="color:cyan"> |-values: [3, 4, 5]</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">psize2 (Choice)</h2></span>



<span style="color:cyan"> |-default: 2</span>



<span style="color:blue"> |-ordered: True</span>



<span style="color:cyan"> |-values: [2, 3]</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">fsize3 (Choice)</h2></span>



<span style="color:cyan"> |-default: 3</span>



<span style="color:blue"> |-ordered: True</span>



<span style="color:cyan"> |-values: [3, 4, 5]</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">psize3 (Choice)</h2></span>



<span style="color:cyan"> |-default: 2</span>



<span style="color:blue"> |-ordered: True</span>



<span style="color:cyan"> |-values: [2, 3]</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">fsize4 (Choice)</h2></span>



<span style="color:cyan"> |-default: 3</span>



<span style="color:blue"> |-ordered: True</span>



<span style="color:cyan"> |-values: [3, 4, 5]</span>



<span style="color:#7E57C2"><h2 style="font-size:16px">psize4 (Choice)</h2></span>



<span style="color:cyan"> |-default: 2</span>



<span style="color:blue"> |-ordered: True</span>



<span style="color:cyan"> |-values: [2, 3]</span>


Perform the search (10 trials, each 10-epoch trial is repeated thrice)


```
tuner_2.search(train_images, train_labels, 
             validation_data=(test_images, test_labels),
             epochs = 10)
```


View details for the best model


```
best_model_2 = tuner_2.get_best_models(num_models=1)[0]

best_hp_2 = tuner_2.get_best_hyperparameters()[0]
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 224, 224, 64)      1088      
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 112, 112, 64)      0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 110, 110, 64)      36928     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 36, 36, 64)        0         
    _________________________________________________________________
    dropout (Dropout)            (None, 36, 36, 64)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 33, 33, 64)        65600     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 11, 11, 64)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 11, 11, 64)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 7, 7, 64)          102464    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 3, 3, 64)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 576)               0         
    _________________________________________________________________
    dense (Dense)                (None, 256)               147712    
    _________________________________________________________________
    dense_1 (Dense)              (None, 7)                 1799      
    =================================================================
    Total params: 355,591
    Trainable params: 355,591
    Non-trainable params: 0
    _________________________________________________________________




---

# **Final Network**

Train a new model using the tuned network configuration for 30 epochs:


```
model = tuner_2.hypermodel.build(best_hp_2)
history = model.fit(train_images, train_labels, epochs=30,
                validation_data=(test_images, test_labels))
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 224, 224, 64)      1088      
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 112, 112, 64)      0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 110, 110, 64)      36928     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 36, 36, 64)        0         
    _________________________________________________________________
    dropout (Dropout)            (None, 36, 36, 64)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 33, 33, 64)        65600     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 11, 11, 64)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 11, 11, 64)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 7, 7, 64)          102464    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 3, 3, 64)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 576)               0         
    _________________________________________________________________
    dense (Dense)                (None, 256)               147712    
    _________________________________________________________________
    dense_1 (Dense)              (None, 7)                 1799      
    =================================================================
    Total params: 355,591
    Trainable params: 355,591
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/30
    96/96 [==============================] - 4s 41ms/step - loss: 1.0477 - accuracy: 0.6688 - val_loss: 0.8435 - val_accuracy: 0.7330
    Epoch 2/30
    96/96 [==============================] - 4s 40ms/step - loss: 0.8733 - accuracy: 0.7078 - val_loss: 0.7577 - val_accuracy: 0.7382
    Epoch 3/30
    96/96 [==============================] - 4s 40ms/step - loss: 0.7697 - accuracy: 0.7287 - val_loss: 0.7026 - val_accuracy: 0.7657
    Epoch 4/30
    96/96 [==============================] - 4s 40ms/step - loss: 0.6941 - accuracy: 0.7549 - val_loss: 0.6445 - val_accuracy: 0.7749
    Epoch 5/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.6209 - accuracy: 0.7795 - val_loss: 0.5799 - val_accuracy: 0.7906
    Epoch 6/30
    96/96 [==============================] - 4s 40ms/step - loss: 0.5756 - accuracy: 0.7935 - val_loss: 0.5181 - val_accuracy: 0.8076
    Epoch 7/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.5140 - accuracy: 0.8122 - val_loss: 0.5468 - val_accuracy: 0.7906
    Epoch 8/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.4732 - accuracy: 0.8315 - val_loss: 0.4940 - val_accuracy: 0.8416
    Epoch 9/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.4441 - accuracy: 0.8334 - val_loss: 0.4141 - val_accuracy: 0.8613
    Epoch 10/30
    96/96 [==============================] - 4s 40ms/step - loss: 0.3638 - accuracy: 0.8658 - val_loss: 0.4700 - val_accuracy: 0.8429
    Epoch 11/30
    96/96 [==============================] - 4s 40ms/step - loss: 0.3452 - accuracy: 0.8688 - val_loss: 0.4523 - val_accuracy: 0.8390
    Epoch 12/30
    96/96 [==============================] - 4s 40ms/step - loss: 0.2898 - accuracy: 0.8917 - val_loss: 0.3885 - val_accuracy: 0.8652
    Epoch 13/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.2740 - accuracy: 0.8963 - val_loss: 0.4263 - val_accuracy: 0.8757
    Epoch 14/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.2515 - accuracy: 0.9035 - val_loss: 0.4804 - val_accuracy: 0.8220
    Epoch 15/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.2332 - accuracy: 0.9169 - val_loss: 0.4285 - val_accuracy: 0.8757
    Epoch 16/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.1991 - accuracy: 0.9287 - val_loss: 0.5097 - val_accuracy: 0.8652
    Epoch 17/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.1929 - accuracy: 0.9359 - val_loss: 0.5533 - val_accuracy: 0.8743
    Epoch 18/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.1864 - accuracy: 0.9332 - val_loss: 0.4026 - val_accuracy: 0.8796
    Epoch 19/30
    96/96 [==============================] - 4s 40ms/step - loss: 0.1519 - accuracy: 0.9421 - val_loss: 0.4567 - val_accuracy: 0.8783
    Epoch 20/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.1430 - accuracy: 0.9470 - val_loss: 0.4967 - val_accuracy: 0.8691
    Epoch 21/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.1608 - accuracy: 0.9421 - val_loss: 0.5270 - val_accuracy: 0.8691
    Epoch 22/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.1241 - accuracy: 0.9542 - val_loss: 0.5301 - val_accuracy: 0.8691
    Epoch 23/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.1346 - accuracy: 0.9503 - val_loss: 0.5279 - val_accuracy: 0.8626
    Epoch 24/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.1168 - accuracy: 0.9568 - val_loss: 0.5361 - val_accuracy: 0.8770
    Epoch 25/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.0843 - accuracy: 0.9709 - val_loss: 0.5911 - val_accuracy: 0.8639
    Epoch 26/30
    96/96 [==============================] - 4s 40ms/step - loss: 0.1138 - accuracy: 0.9581 - val_loss: 0.6072 - val_accuracy: 0.8717
    Epoch 27/30
    96/96 [==============================] - 4s 40ms/step - loss: 0.0981 - accuracy: 0.9624 - val_loss: 0.6422 - val_accuracy: 0.8730
    Epoch 28/30
    96/96 [==============================] - 4s 40ms/step - loss: 0.0931 - accuracy: 0.9692 - val_loss: 0.5268 - val_accuracy: 0.8796
    Epoch 29/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.0724 - accuracy: 0.9735 - val_loss: 0.6417 - val_accuracy: 0.8678
    Epoch 30/30
    96/96 [==============================] - 4s 39ms/step - loss: 0.1197 - accuracy: 0.9604 - val_loss: 0.5626 - val_accuracy: 0.8665


Visualize the accuracy and loss as a function of number of epochs.


```
def plot_history(model):
  plt.figure(figsize=(15,5))
  plt.subplot(1,2,1)
  plt.plot(model.history['accuracy'])
  plt.plot(model.history['val_accuracy'])
  plt.title('Accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')

  plt.subplot(1,2,2)
  plt.plot(model.history['loss'])
  plt.plot(model.history['val_loss'])
  plt.title('Loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper right')
  
  plt.show()

plot_history(history)
```


![png](ajn170130_mars_CCN_files/ajn170130_mars_CCN_30_0.png)


Train the final model for 18 epochs for optimal balance of high accuracy and low loss


```
model = tuner_2.hypermodel.build(best_hp_2)
history = model.fit(train_images, train_labels, epochs=18,
                validation_data=(test_images, test_labels))

plot_history(history)
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 224, 224, 64)      1088      
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 112, 112, 64)      0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 110, 110, 64)      36928     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 36, 36, 64)        0         
    _________________________________________________________________
    dropout (Dropout)            (None, 36, 36, 64)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 33, 33, 64)        65600     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 11, 11, 64)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 11, 11, 64)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 7, 7, 64)          102464    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 3, 3, 64)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 576)               0         
    _________________________________________________________________
    dense (Dense)                (None, 256)               147712    
    _________________________________________________________________
    dense_1 (Dense)              (None, 7)                 1799      
    =================================================================
    Total params: 355,591
    Trainable params: 355,591
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/18
    96/96 [==============================] - 4s 41ms/step - loss: 1.0668 - accuracy: 0.6623 - val_loss: 0.9667 - val_accuracy: 0.7225
    Epoch 2/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.8522 - accuracy: 0.7084 - val_loss: 0.7815 - val_accuracy: 0.7448
    Epoch 3/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.7768 - accuracy: 0.7186 - val_loss: 0.6460 - val_accuracy: 0.7723
    Epoch 4/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.6604 - accuracy: 0.7628 - val_loss: 0.6437 - val_accuracy: 0.7709
    Epoch 5/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.5845 - accuracy: 0.7902 - val_loss: 0.5174 - val_accuracy: 0.8168
    Epoch 6/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.5098 - accuracy: 0.8086 - val_loss: 0.5283 - val_accuracy: 0.8154
    Epoch 7/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.4467 - accuracy: 0.8354 - val_loss: 0.4990 - val_accuracy: 0.8298
    Epoch 8/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.4026 - accuracy: 0.8527 - val_loss: 0.5064 - val_accuracy: 0.8102
    Epoch 9/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.3881 - accuracy: 0.8593 - val_loss: 0.4451 - val_accuracy: 0.8390
    Epoch 10/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.3530 - accuracy: 0.8675 - val_loss: 0.4389 - val_accuracy: 0.8586
    Epoch 11/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.3322 - accuracy: 0.8838 - val_loss: 0.4790 - val_accuracy: 0.8521
    Epoch 12/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.2876 - accuracy: 0.8979 - val_loss: 0.4592 - val_accuracy: 0.8390
    Epoch 13/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.2696 - accuracy: 0.9028 - val_loss: 0.4971 - val_accuracy: 0.8442
    Epoch 14/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.2562 - accuracy: 0.9094 - val_loss: 0.4256 - val_accuracy: 0.8508
    Epoch 15/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.2068 - accuracy: 0.9267 - val_loss: 0.5072 - val_accuracy: 0.8613
    Epoch 16/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.2224 - accuracy: 0.9224 - val_loss: 0.4706 - val_accuracy: 0.8573
    Epoch 17/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.1929 - accuracy: 0.9296 - val_loss: 0.4400 - val_accuracy: 0.8717
    Epoch 18/18
    96/96 [==============================] - 4s 40ms/step - loss: 0.1547 - accuracy: 0.9440 - val_loss: 0.4602 - val_accuracy: 0.8743



![png](ajn170130_mars_CCN_files/ajn170130_mars_CCN_32_1.png)




---

## **Predictions**

Prediction plotting functions from https://www.tensorflow.org/tutorials/keras/classification


```
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary_r)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_map[predicted_label],
                                100*np.max(predictions_array),
                                class_map[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(7))
  plt.yticks([])
  thisplot = plt.bar(range(7), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  

```

Make predictions on the test set and visualize:


```
predictions = model.predict(test_images)

test_images = test_images.reshape((test_images.shape[0],227,227))

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 10
num_cols = 2
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
```


![png](ajn170130_mars_CCN_files/ajn170130_mars_CCN_36_0.png)

