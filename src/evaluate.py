'''
Created on Jun 13, 2018

@author: Inthuch Therdchanakul
'''
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from keras.models import model_from_json, load_model
from sklearn.metrics import confusion_matrix
import os

if __name__ == '__main__':
    #classify each image in the validation set
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']
#     json_file = open('../PretrainLSTM.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     # load weights into new model
#     loaded_model.load_weights("../PreTraineLSTMWeight.h5")
    loaded_model = load_model('../LSTM_256_1024.h5')
    print("Loaded model from disk")
    
    # evaluate loaded model on test data
    loaded_model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    path = '../dataFromPython3/validation/'
    Accurate = 0
    counter = 0
    labelsValues = []
    PredictedValues = []
    test_data = np.load(open('../bottleneck_features_test.npy', 'rb'))
    counterImage = 0
    for emotion in emotions:
        for file in os.listdir(path+emotion):
            if 'png' in file: # and counterImage< 689:
                image = test_data[counterImage]
                image = image.reshape(1, len(image), len(image)*512)
                label = emotions.index(emotion)
                pred = loaded_model.predict(image)
                pred1 = list(pred[0])
                max_value = max(pred1)
                max_index = pred1.index(max_value)
                p = max_index
                if p == label:
                    Accurate += 1
                labelsValues.append(label)

                PredictedValues.append(p)

#                 if emotion == 'sad':
#                     print(file)
#                     print(p)

                counter +=1
                counterImage +=1

#                 print(counterImage)
    acc = float(Accurate)/float(counter)
    results = confusion_matrix(labelsValues, PredictedValues)
    print(results)
    print(acc)
    
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprise']
    lookup = {0: "angry", 1: "disgusted", 2: 'fearful',3: "happy", 4: 'sad', 5: "surprise"}  # , 6:"reassured"}
    y_true = pd.Series([lookup[_] for _ in labelsValues])  # np.random.random_integers(0, 5, size=100)])
    y_pred = pd.Series([lookup[_] for _ in PredictedValues])  # np.random.random_integers(0, 5, size=100)])

    '''print('positive: ' + str(overallCounter))
    print('total: ' + str(numIm))
    print('accuracy: ' + str(acc))'''
    pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted']).apply(lambda r: 100.0 * r / r.sum())
    conf = confusion_matrix(y_true, y_pred)

    #######################################################################################
    lookup = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4:  "sad",
              5: "surprise"}  # , 6:"reassured"}
    # lookup = {0: 'Angry', 1: 'Disgust', 2: 'happy', 3: 'neutral', 4: 'surprised', 5: 'Sad', 6: 'fearful'}
    y_true = pd.Series([lookup[_] for _ in labelsValues])  # np.random.random_integers(0, 5, size=100)])
    y_pred = pd.Series([lookup[_] for _ in PredictedValues])  # np.random.random_integers(0, 5, size=100)])

    res = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], normalize='index')

    ###########################################added CONFUSION WITHOU PERCENTAGE
    conf = confusion_matrix(y_true, y_pred)
    conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
    y_true1 = np.array(y_true)
    i=0
    y_true1.tofile('../y_trueCNNCohenCohen'+str(i)+'.csv', sep=',')
    y_pred1 = np.array(y_pred)
    y_pred1.tofile('../y_predCNNCohenCohen'+str(i)+'.csv', sep=',')

    plt.imshow(conf, interpolation='nearest', cmap='Reds')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(emotions))
    plt.xticks(tick_marks, emotions, rotation=45)
    plt.yticks(tick_marks, emotions)

    fmt = '.3f'
    thresh = conf.max() / 2.
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
        plt.text(j, i, format(conf[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('../confusion_matrix_percent_LSTM.png', format='png')
    plt.show()

    #######################################################################################
    conf = results
    norm_conf = []
    for i in conf:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap="Reds",
                    interpolation='nearest')

    width, height = conf.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    plt.xticks(range(width), emotions)  # alphabet[:width])
    plt.yticks(range(height), emotions)  # alphabet[:height])
#     plt.show()
    plt.savefig('../confusion_matrixLSTM.png', format='png')

