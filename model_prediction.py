"""

Model Prediction using Neural Network

"""


import features
import pickle
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class Parameters:

    def __init__(self,file_name,feature_name,lower_limit,upper_limit,test_size,random_seed):

        self.model_features = features.Model_Features(file_name,feature_name,lower_limit,upper_limit)
        self.x_train, self.x_test, self.y_train, self.y_test = self.model_features.training_validation_split(test_size_split=test_size,random_seed=random_seed)

class Model(Parameters):

    def __init__(self,file_name,feature_name,lower_limit,upper_limit,test_size,random_seed):
        
        super().__init__(file_name,feature_name,lower_limit,upper_limit,test_size,random_seed)

        self.classifier = Sequential()
        self.scalar = StandardScaler()
        self.train_x_ = self.scalar.fit_transform(self.x_train)
        self.test_x_ = self.scalar.transform(self.x_test)

    def build_the_model(self,inputs,total_class,activation):

        self.classifier.add(Dense(
            units=144, kernel_initializer='he_uniform', activation='relu', input_dim=inputs))
        self.classifier.add(
            Dense(units=400, kernel_initializer='he_uniform', activation='relu'))
        self.classifier.add(
            Dense(units=400, kernel_initializer='he_uniform', activation='relu'))
        self.classifier.add(
            Dense(units=272, kernel_initializer='he_uniform', activation='relu'))
        self.classifier.add(
            Dense(units=400, kernel_initializer='he_uniform', activation='relu'))
        self.classifier.add(
            Dense(units=400, kernel_initializer='he_uniform', activation='relu'))
        self.classifier.add(
            Dense(units=400, kernel_initializer='he_uniform', activation='relu'))
        self.classifier.add(
            Dense(units=400, kernel_initializer='he_uniform', activation='relu'))
        self.classifier.add(
            Dense(units=272, kernel_initializer='he_uniform', activation='relu'))
        self.classifier.add(
            Dense(units=total_class, kernel_initializer='glorot_uniform', activation=activation))

    def compile_nn(self,inputs,total_class,activation,batch,epochs,loss,metrics):

        self.build_the_model(inputs,total_class,activation)

        self.classifier.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001), loss=loss, metrics=[metrics])
        self.model_history = self.classifier.fit(
            self.train_x_, self.y_train, batch_size=batch, epochs=epochs)

    def test_model_accuracy(self,inputs,total_class,activation,batch,epochs,loss,metrics):

        self.compile_nn(inputs,total_class,activation,batch,epochs,loss,metrics)
        model_predict = self.classifier.predict(self.test_x_)
        #training_result = self.classifier.predict(self.train_x_)
        y_pred = (model_predict > 0.5)
        #y_pred_training = (training_result > 0.5)
        return accuracy_score(self.y_test, y_pred) , classification_report(self.y_test, y_pred), self.classifier, self.scalar , self.x_train.columns


if __name__ == '__main__':
    m = Model(file_name='loan_prediction.csv',feature_name='Loan_Status',lower_limit=1,upper_limit=13,test_size=0.2,random_seed=100)
    score, report, model , scalar_model , training_features = m.test_model_accuracy(inputs=14,total_class=1,
                                            activation='sigmoid',batch=25,epochs=200,
                                            loss='binary_crossentropy',metrics='accuracy')
    print(score)
    print(report)
    column_name = [feature for feature in training_features]


    

    pickle.dump(scalar_model, open('scaler.pkl','wb'))
    if score > 0.80:
        model.save('loan_prediction.h5')
        with open('model_summary.txt','w') as ms:
            ms.write(report)
        ms.close()
    else:
        print('model overfitting')

    with open('features.txt','w') as feature_names:
        for word in column_name:
            feature_names.write(str(word) + "\n")
    feature_names.close()
        
    
