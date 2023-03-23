
"""
Neural Network Hyper parameter tuning

"""

import features
import tensorflow as tf
import keras_tuner as kt
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler

class Parameters:

  def __init__(self,file_name,feature_name,lower_limit,upper_limit,test_size,random_seed):
    self.model_features = features.Model_Features(file_name,feature_name,lower_limit,upper_limit)
    self.x_train, self.x_test, self.y_train, self.y_test = self.model_features.training_validation_split(test_size_split=test_size,random_seed=random_seed)

  def training_features(self):
    return self.x_train, self.x_test, self.y_train, self.y_test
     
param = Parameters(file_name='loan_prediction.csv',feature_name='Loan_Status',lower_limit=1,upper_limit=13,test_size=0.2,random_seed=100)
train_x , test_x , train_y , test_y = param.training_features()



train_x_scl = StandardScaler().fit_transform(train_x)
test_x_scl = StandardScaler().transform(test_x)

def build_model(hp):
    model = Sequential()
    for i in range(hp.Int('num_layers' , 2, 20)):
      model.add(Dense(hp.Int('units_'+str(i),min_value = 16,max_value = 512,step = 128),
              activation='relu'))
    
    model.add(Dense(1,activation='sigmoid'))
    model.compile(
      optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate' , [1e-2,1e-3,1e-4])),
      loss = 'binary_crossentropy',
      metrics = ['accuracy'])
    return model
    
def tuner():
    tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    project_name = 'kears_tuner')
    tuner.search(train_x_scl , train_y, epochs=20, validation_data=(test_x_scl,test_y))
    return tuner.results_summary()

hp_tuner = tuner()
print(hp_tuner)


        