import grace_covid_model
import evaluation
import numpy as np
import tensorflow as tf

def train():
    x = np.load('data/x_train.npy')
    y = np.load('data/y_train.npy')
    y_onehot = tf.keras.utils.to_categorical(y, 4) #normal, bacterial, viral, COVID-19
    
    #CovidNet(input_shape, num_classes=3):
    model = grace_covid_model.CovidNet(input_size=(224, 224, 3)).CovidNet() #keras_model.keras_model_build()
    # model.summary()
    # Implementtation Details: 
    '''
    learning_rate = 2e-5, number of epochs = 100, batch size = 8, 
    factor = 0.7, patience = 5
    '''
    loss_func = tf.keras.losses.categorical_crossentropy # onehot encoding을 진행 한 데이터인 경우 categorical(multiple) Classification일 때
    optm = tf.keras.optimizers.Adam(lr = 0.00002)
    metrics = tf.keras.metrics.Accuracy()

    model.compile(optimizer = optm, loss = loss_func, metrics = metrics)

    num_epochs = 100
    batch_size = 8

    model.fit(x, y_onehot, batch_size = batch_size, epochs = num_epochs, shuffle = True, verbose=1)
    #verbose는 학습 중 출력되는 문구를 설정하는 것으로, 주피터노트북(Jupyter Notebook)을 사용할 때는 verbose=1은 progress bar
    # reference: https://datascienceschool.net/view-notebook/51e147088d474fe1bf32e394394eaea7/

    model.save('model.h5')
    #학습된 모델 저장하기: 모델은 크게 모델 아키텍처와 모델 가중치로 구성됩니다. 모델 아키텍처는 모델이 어떤 층으로 어떻게 쌓여있는 지에 대한 모델 구성이 정의되어 있고, 모델 가중치는 처음에는 임의의 값으로 초기화되어 있지만, 훈련셋으로 학습하면서 갱신됩니다. 학습된 모델을 저장한다는 말은 ‘모델 아키텍처’와 ‘모델 가중치’를 저장한다는 말입니다. 케라스에서는 save() 함수 하나로 ‘모델 아키텍처’와 ‘모델 가중치’를 ‘h5’파일 형식으로 모두 저장할 수 있습니다.
    # reference: https://tykimos.github.io/2017/06/10/Model_Save_Load/

    y_pred = model.predict(x)

    #그리고 train 성능 평가. 
    evaluation.confusion_matrix_info(np.argmax(y_onehot, axis=1), np.argmax(y_pred, axis=1),title='confusion_matrix_train')


