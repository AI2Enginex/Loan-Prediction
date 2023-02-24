
"""

Making Data Ready for Training
Preprocessing stage 
seprating dependent and independent variables

"""

import warnings
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore", category=FutureWarning)


class Load_data:

    def __init__(self,file_name,feature_name,lower_limit,upper_limit):                         
        self.loan = pd.read_csv(file_name, usecols=range(lower_limit,upper_limit))
        self.loan[feature_name] = LabelEncoder().fit_transform(self.loan[feature_name])

    def remove_null_values(self):
        for cols in self.loan.columns:
            if self.loan[cols].dtypes == 'object' and self.loan[cols].isnull().sum() > 0:
                self.loan[cols] = self.loan[cols].fillna(
                    self.loan[cols].mode()[0])
            elif self.loan[cols].dtypes != 'object' and self.loan[cols].isnull().sum() > 0:
                self.loan[cols] = self.loan[cols].fillna(
                    self.loan[cols].median())

    def detect_outliers(self):

        self.remove_null_values()
        features_ = [cols for cols in self.loan.columns if self.loan[cols].dtypes !=
                     'object' and len(self.loan[cols].unique()) > 10]
        iso_f = IsolationForest().fit(self.loan[features_])
        pred = iso_f.predict(self.loan[features_])
        self.loan['outliers'] = pred

    def create_dummies(self):

        self.detect_outliers()
        cat_features = [
            cols for cols in self.loan.columns if self.loan[cols].dtypes == 'object']
        self.data_ = pd.concat([self.loan, pd.get_dummies(
            self.loan[cat_features], drop_first=True)], axis=1)


class Training_split(Load_data):

    def __init__(self,file_name,feature_name,lower_limit,upper_limit):
        super().__init__(file_name,feature_name,lower_limit,upper_limit)
        self.create_dummies()
        self.loan_ = self.data_
        self.feature = feature_name

    def remove_categorical_features(self):
        categorical_features = [
            cols for cols in self.loan_.columns if self.loan_[cols].dtypes == 'object']
        self.loan_ = self.loan_.drop(categorical_features, axis=1)

    def removing_outliers(self):

        self.remove_categorical_features()
        self.loan_ = self.loan_[self.loan_['outliers'] == 1]
        self.loan_ = self.loan_.drop('outliers', axis=1)

    def random_over_sampling(self):

        self.removing_outliers()
        self.x = self.loan_.drop(self.feature, axis=1)
        self.y = self.loan_[self.feature]
        count_values = Counter(self.y)
        for key in count_values.keys():
            if count_values[key] != count_values[key+1]:
                self.x, self.y = RandomOverSampler().fit_resample(self.x, self.y)
            else:
                pass


class Model_Features(Training_split):

    def __init__(self,file_name,feature_name,lower_limit,upper_limit):
        super().__init__(file_name,feature_name,lower_limit,upper_limit)
        self.random_over_sampling()
        self.independent_features = self.x
        self.dependent_features = self.y

    def training_validation_split(self,test_size_split,random_seed):

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.independent_features, self.dependent_features,
                                                                                test_size=test_size_split,
                                                                                random_state=random_seed)

        return self.train_x, self.test_x, self.train_y, self.test_y




if __name__ == "__main__":
   
   pass
