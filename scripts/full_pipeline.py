from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class BedroomImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        imputer = SimpleImputer(strategy="median")
        X['total_bedrooms'] = imputer.fit_transform(X[['total_bedrooms']]) #--> [['...']] returns the "total_bedrooms" column as a Dataframe rather than Series(when used a single square-bracket)
        # We didn't used SimpleImputer directly because this takes a numpy array as argument. so to ease out, we put an another abstraction layer
        return X

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class AttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room 
    
    def fit(self, X, y=None):
        return self # nothing else to do 

    def transform(self, X, y=None):
        X = X.copy()  # to avoid SettingWithCopyWarning
        X['rooms_per_household'] = X['total_rooms'] / X['households']
        X['population_per_household'] = X['population'] / X['households']
        if self.add_bedrooms_per_room:
            X['bedrooms_per_room'] = X['total_bedrooms'] / X['total_rooms']
        return X
    

pipeline_v1= Pipeline([
    ('imputer', BedroomImputer()),
    ('attribute_adder', AttributesAdder(True)),
    ('feature_scaler', StandardScaler()) # -> StandardScaler() is the one that gives output as a numpy array
])

pipeline_v2 = ColumnTransformer([
    ('numerical_pipe', pipeline_v1, list(features.drop(['ocean_proximity'], axis=1))), # type: ignore
    ('categorical_pipe', OneHotEncoder(), ["ocean_proximity"])
])

complete_pipeline = Pipeline([
    {'partial_pipeline', pipeline_v2},
    {'prdeictor', LinearRegression()}
])

if __name__!="__main__":
    print("\n" + "#"*8 + "  Script is written by BHAVIK OSTWAL  " + '#'*8 + '\n')