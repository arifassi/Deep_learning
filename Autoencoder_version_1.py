from os import name
#In[]

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import silhouette_samples, silhouette_score
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
import keras_tuner



#In[]
encoders= []
keras.backend.clear_session()

#In[]
from tensorflow.keras.utils import to_categorical

n_classes=10
division= 10
acc=[]
loss=[]
rations=[]
for k in range(1,10):
    (X_train, y_train), (X_test, y_test),  = mnist.load_data()
    # splitting labeled and unlabeled data
    # ratio of labeled data to total dataset

    ratio=k/division

    X_trainun, X_train, y_trainun, y_train = train_test_split(X_train, y_train, test_size=ratio, stratify=y_train)
    # splitting the validation set

    val_size = .1

    X_trainun, X_valun, y_trainun, y_valun = train_test_split(X_trainun, y_trainun, test_size=val_size, stratify=y_trainun)


    X_trainun = X_trainun/255
    X_valun = X_valun/255
    X_test = X_test/255

    # Define model architecture
    def build_model(hp):
        autoencoder=Sequential()

        autoencoder.add(layers.Input(shape=(28,28)))
        autoencoder.add(layers.Reshape((28,28,1)))  #reshape
        autoencoder.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
        autoencoder.add(layers.MaxPooling2D((2, 2), padding='same'))
        autoencoder.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
        autoencoder.add(layers.MaxPooling2D((2, 2), padding='same'))
        autoencoder.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
        autoencoder.add(layers.MaxPooling2D((2, 2), padding='same', name='encoded'))

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        autoencoder.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
        autoencoder.add(layers.UpSampling2D((2, 2)))
        autoencoder.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
        autoencoder.add(layers.UpSampling2D((2, 2)))
        autoencoder.add(layers.Conv2D(16, (3, 3), activation='relu'))
        autoencoder.add(layers.UpSampling2D((2, 2)))
        autoencoder.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoded'))

        learning_rate = hp.Float("learning_rate", min_value=1e-3, max_value=1, sampling="log")
        autoencoder.compile(optimizer=SGD(learning_rate = learning_rate), loss='binary_crossentropy')
        return autoencoder

    build_model(keras_tuner.HyperParameters())
    tuner_autoencoder= keras_tuner.GridSearch(build_model, max_trials = 10,executions_per_trial=2, objective=keras_tuner.Objective('val_loss', 'min'), overwrite=True)
    #tuner_autoencoder= keras_tuner.GridSearch(build_model, max_trials = 10,executions_per_trial=2, objective=keras_tuner.Objective('val_loss', 'min'), directory=f"autoencoder{ratio}")
    tuner_autoencoder.search(X_trainun, X_trainun, epochs=5, validation_data=(X_valun, X_valun))

    # Get the top 2 hyperparameters.
    best_hps = tuner_autoencoder.get_best_hyperparameters(5)

    # Build the model with the best hp.
    autoencoder = build_model(best_hps[0])

    # Fit with the entire dataset.
    x_all = np.concatenate((X_trainun, X_valun))
    y_all = np.concatenate((y_trainun, y_valun))
    autoencoder.fit(x=X_trainun,y=X_trainun, epochs=5)


    # Extract the encoder from the model
    encoder = keras.Model(
        inputs=autoencoder.inputs,
        outputs=autoencoder.get_layer(name='encoded').output,
    )
    encoders= np.append(encoders, encoder)

#In[]
#Import confusion matrix 

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#In[]

n_classes=10
division= 10
acc=[]
loss=[]
rations=[]  
for k in range(1,10):
    (X_train, y_train), (X_test, y_test),  = mnist.load_data()
    # splitting labeled and unlabeled data
    # ratio of labeled data to total dataset

    ratio=k/division

    X_trainun, X_train, y_trainun, y_train = train_test_split(X_train, y_train, test_size=ratio, stratify=y_train)
    # splitting the validation set

    val_size = .1

    X_trainun, X_valun, y_trainun, y_valun = train_test_split(X_trainun, y_trainun, test_size=val_size, stratify=y_trainun)


    X_trainun = X_trainun/255
    X_valun = X_valun/255
    X_test = X_test/255

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, stratify=y_train)

    y_train_cat = to_categorical(y_train, n_classes)
    y_val_cat = to_categorical(y_val, n_classes)
    y_test_cat = to_categorical(y_test, n_classes)

    encoded_train = encoders[k-1].predict(X_train)
    encoded_val = encoders[k-1].predict(X_val)

    def build_classifier(hp):
        classifier=Sequential()

        classifier.add(layers.Input(shape=(4,4,8)))
        classifier.add(layers.Flatten())
        #classifier.add(layers.Dense(n_classes, activation='sigmoid'))
        classifier.add(layers.Dense(n_classes, activation='softmax'))
        learning_rate = hp.Float("learning_rate", min_value=1e-3, max_value=1, sampling="log")
        classifier.compile(optimizer=SGD(learning_rate = learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        #classifier.compile(optimizer=SGD(learning_rate = learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        return classifier 
    


    build_classifier(keras_tuner.HyperParameters())
    tuner= keras_tuner.GridSearch(build_classifier, max_trials = 20,executions_per_trial=3, objective=keras_tuner.Objective('val_accuracy', 'max'), overwrite=True)
    #tuner= keras_tuner.GridSearch(build_classifier, max_trials = 20,executions_per_trial=3, objective=keras_tuner.Objective('val_accuracy', 'max'), directory= f"binari-classifier{ratio}")
    tuner= keras_tuner.GridSearch(build_classifier, max_trials = 20,executions_per_trial=3, objective=keras_tuner.Objective('val_accuracy', 'max'), directory= f"classifier{ratio}")
    #tuner.search(encoded_train, y_train_cat, epochs=5, validation_data=(encoded_val, y_val_cat))

    # Get the top 2 hyperparameters.
    best_hps = tuner.get_best_hyperparameters(5)

    # Build the model with the best hp.
    classifier = build_classifier(best_hps[0])
    # Fit with the entire dataset.
    encoded_all = np.concatenate((encoded_train, encoded_val))
    y_all = np.concatenate((y_train_cat, y_val_cat))
    classifier.fit(x= encoded_all, y=y_all, epochs=5)


    # Evaluate the model on the test set

    encoded_test= encoders[k-1].predict(X_test)

    test_loss, test_acc = classifier.evaluate(encoded_test, y_test_cat)
    loss=np.append(loss,test_loss)
    acc=np.append(acc,test_acc)
    rations=np.append(rations, k/division)
    print(acc)
    print(loss)
    print(rations)
    print()
    print('Test\t',k, ' Loss:\t', test_loss)
    print('Test\t',k, 'Accuracy:\t', test_acc)

    # confusion matrix

    predictions = classifier.predict(encoded_test)
    pred= np.argmax(predictions, axis=1)
    print(pred)
    print(y_test)
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - Classifier with Autoencoder, ratio ={ratio}")
    plt.savefig(f"Confusion Matrix - Classifier with Autoencoder, ratio ={ratio}.png", bbox_inches="tight",
            pad_inches=0.3, transparent=False)
    #plt.title(f"Confusion Matrix - Binary Classifier with Autoencoder, ratio ={ratio}")
    #plt.savefig(f"Confusion Matrix - Binary Classifier with Autoencoder, ratio ={ratio}.png", bbox_inches="tight",
     #       pad_inches=0.3, transparent=False)
    plt.show()


#In[]
print()
plt.figure
plt.scatter(rations, acc)
plt.xlabel('Ratio')
plt.ylabel('Accuracy')
plt.title('Autoencoders ')
plt.savefig(f"Classifier with autoencoder accuracy.png", bbox_inches="tight",
            pad_inches=0.3, transparent=False)
#plt.savefig(f"Binary Classifier with autoencoder accuracy.png", bbox_inches="tight",
#            pad_inches=0.3, transparent=False)
plt.show()




