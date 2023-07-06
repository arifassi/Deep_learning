
# In[3]:


#standard imports
import numpy as np
import matplotlib.pyplot as plt

#import tensorflow and keras
import tensorflow as tf
from tensorflow import keras


#function for fast generation of training and test set from data
from sklearn.model_selection import train_test_split


# Convert labels to one-hot encoded vectors

#transform integer label into one-hot encodings
from tensorflow.keras.utils import to_categorical


#a few more imports for readability

#import generic model class
from tensorflow.keras import Model, Input

# Faccio il Grid Search del'iperparametro da stimare: il learning rate 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten , Input , Reshape , Dropout
#import the types of layes we need
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten , Input , Reshape , Dropout

from tensorflow.keras.optimizers import Adam, SGD
import keras_tuner


#Confusion MATRIX
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[4]:

# **Loading mnist**
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[5]:


#check shape of the training set
print('TRAINING SET:\t',X_train.shape)

#check shape of the test set
print('TEST SET:\t',    X_test.shape)


# In[6]:

#show first image contained in the training data, with its label

plt.imshow(X_train[0], cmap='gray')
plt.title('NUMBER {}'.format(y_train[0]), fontsize=15)
plt.axis('off'); 
#do not show spines




# In[8]:


# show the first 10 images in the training set

fig, axs = plt.subplots(2,5,figsize=(12,4))

for i in range(10):
    
    ax = axs.ravel()[i]
    
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title('NUMBER {}'.format(y_train[i]), fontsize=13)
    ax.axis('off')

# In[9]:

plt.hlines(6000, -1, 10, linestyles='--')

plt.hist(y_train, bins=np.linspace(-.5,9.5,11), rwidth=.8)
plt.xticks(range(10))



# In[10]:
# splitting labeled and unlabeled data
# ratio of labeled data to total dataset
n_classes=10
division= 1000
acc=[]
loss=[]
rations=[]
for k in range(1,10):
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    ratio = k/division
    X_train, X_trainun, y_train, y_trainun = train_test_split(X_train, y_train, test_size=1-ratio, stratify=y_train)

    val_size = .2 #use 10% of the labeled samples as validation set

#generate training and test set with stratification
#this means that we will find the same proportion of classes bot in the validation and in the training set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, stratify=y_train)


    #rescale pixel values to the range (0,1)
    X_train = X_train/255
    X_val = X_val/255
    X_test = X_test/255

    y_train_cat = to_categorical(y_train, n_classes)
    y_val_cat = to_categorical(y_val, n_classes)
    y_test_cat = to_categorical(y_test, n_classes)
    keras.backend.clear_session()


    def build_model(hp):
        model = Sequential()
        model.add(Input(shape=(28,28))) #28 x 28 BW 
        model.add(Reshape((28,28,1)))  #reshape 
        #model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        #model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(Flatten())
        #model.add(Dense(n_classes, activation='sigmoid'))
        model.add(Dense(n_classes, activation='softmax'))
        learning_rate = hp.Float("learning_rate", min_value=1e-3, max_value=1, sampling="log")
        model.compile(optimizer=SGD(learning_rate = learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    build_model(keras_tuner.HyperParameters())
    #tuner= keras_tuner.GridSearch(build_model, max_trials = 10,executions_per_trial=3, objective=keras_tuner.Objective('val_accuracy', 'max'), directory=f"binari-cnn{ratio}")
    tuner= keras_tuner.GridSearch(build_model, max_trials = 20,executions_per_trial=3, objective=keras_tuner.Objective('val_accuracy', 'max'), directory=f"cnn{ratio}")
    tuner.search(X_train, y_train_cat, epochs=5, validation_data=(X_val, y_val_cat))

# Get the top 2 hyperparameters.
    best_hps = tuner.get_best_hyperparameters(5)

# Build the model with the best hp.
    model = build_model(best_hps[0])
# Fit with the entire dataset.
    x_all = np.concatenate((X_train, X_val))
    y_all = np.concatenate((y_train_cat, y_val_cat))
    model.fit(x=x_all, y=y_all, epochs=5)


# Evaluate the model on the test set

    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    loss=np.append(loss,test_loss)
    acc=np.append(acc,test_acc)
    rations=np.append(rations, k/division)
    print(acc)
    print(rations)
    print()
    print('Test\t',k, ' Loss:\t', test_loss)
    print('Test\t',k, 'Accuracy:\t', test_acc)
 

# In[10]:
# splitting labeled and unlabeled data
# ratio of labeled data to total dataset
n_classes=10
division= 100
for k in range(1,10):
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    ratio = k/division
    X_train, X_trainun, y_train, y_trainun = train_test_split(X_train, y_train, test_size=1-ratio, stratify=y_train)

    val_size = .1 #use 10% of the labeled samples as validation set

#generate training and test set with stratification
#this means that we will find the same proportion of classes bot in the validation and in the training set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, stratify=y_train)


    #rescale pixel values to the range (0,1)
    X_train = X_train/255
    X_val = X_val/255
    X_test = X_test/255

    y_train_cat = to_categorical(y_train, n_classes)
    y_val_cat = to_categorical(y_val, n_classes)
    y_test_cat = to_categorical(y_test, n_classes)
    keras.backend.clear_session()


    def build_model(hp):
        model = Sequential()
        model.add(Input(shape=(28,28))) #28 x 28 BW 
        model.add(Reshape((28,28,1)))  #reshape 
        #model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        #model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(n_classes, activation='softmax'))
        learning_rate = hp.Float("learning_rate", min_value=1e-3, max_value=1, sampling="log")
        model.compile(optimizer=SGD(learning_rate = learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    build_model(keras_tuner.HyperParameters())
    tuner= keras_tuner.GridSearch(build_model, max_trials = 20,executions_per_trial=3, objective=keras_tuner.Objective('val_accuracy', 'max'), directory=f"cnn{ratio}")
    tuner.search(X_train, y_train_cat, epochs=5, validation_data=(X_val, y_val_cat))

# Get the top 2 hyperparameters.
    best_hps = tuner.get_best_hyperparameters(5)

# Build the model with the best hp.
    model = build_model(best_hps[0])
# Fit with the entire dataset.
    x_all = np.concatenate((X_train, X_val))
    y_all = np.concatenate((y_train_cat, y_val_cat))
    model.fit(x=x_all, y=y_all, epochs=5)


# Evaluate the model on the test set

    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    loss=np.append(loss,test_loss)
    acc=np.append(acc,test_acc)
    rations=np.append(rations, k/division)
    print(acc)
    print(rations)
    print()
    print('Test\t',k, ' Loss:\t', test_loss)
    print('Test\t',k, 'Accuracy:\t', test_acc)


# In[10]:
# splitting labeled and unlabeled data
# ratio of labeled data to total dataset
n_classes=10
division= 10
for k in range(1,10):
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    ratio = k/division
    X_train, X_trainun, y_train, y_trainun = train_test_split(X_train, y_train, test_size=1-ratio, stratify=y_train)

    val_size = .1 #use 10% of the labeled samples as validation set

#generate training and test set with stratification
#this means that we will find the same proportion of classes bot in the validation and in the training set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, stratify=y_train)


    #rescale pixel values to the range (0,1)
    X_train = X_train/255
    X_val = X_val/255
    X_test = X_test/255

    y_train_cat = to_categorical(y_train, n_classes)
    y_val_cat = to_categorical(y_val, n_classes)
    y_test_cat = to_categorical(y_test, n_classes)
    keras.backend.clear_session()


    def build_model(hp):
        model = Sequential()
        model.add(Input(shape=(28,28))) #28 x 28 BW 
        model.add(Reshape((28,28,1)))  #reshape 
        #model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        #model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(n_classes, activation='softmax'))
        learning_rate = hp.Float("learning_rate", min_value=1e-3, max_value=1, sampling="log")
        model.compile(optimizer=SGD(learning_rate = learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    build_model(keras_tuner.HyperParameters())
    tuner= keras_tuner.GridSearch(build_model, max_trials = 20,executions_per_trial=3, objective=keras_tuner.Objective('val_accuracy', 'max'), directory=f"cnn{ratio}")
    tuner.search(X_train, y_train_cat, epochs=5, validation_data=(X_val, y_val_cat))

# Get the top 2 hyperparameters.
    best_hps = tuner.get_best_hyperparameters(5)

# Build the model with the best hp.
    model = build_model(best_hps[0])
# Fit with the entire dataset.
    x_all = np.concatenate((X_train, X_val))
    y_all = np.concatenate((y_train_cat, y_val_cat))
    model.fit(x=x_all, y=y_all, epochs=5)


# Evaluate the model on the test set

    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    loss=np.append(loss,test_loss)
    acc=np.append(acc,test_acc)
    rations=np.append(rations, k/division)
    print(acc)
    print(rations)
    print()
    print('Test\t',k, ' Loss:\t', test_loss)
    print('Test\t',k, 'Accuracy:\t', test_acc)



#In[]
print(acc)
print(rations)
print(loss)
plt.figure 
#plt.scatter(rations, acc, color= 'blue') 
plt.scatter(rations, loss, color='red')
plt.xlabel('Ratio')
plt.ylabel('Loss')
plt.title('Convolutional NN trained normally')
plt.savefig(f"CNN-classic accuracy.png", bbox_inches="tight",
           pad_inches=0.3, transparent=False)
#plt.savefig(f"Binary CNN-classic accuracy.png", bbox_inches="tight",
#            pad_inches=0.3, transparent=False)
plt.show()


# %%
