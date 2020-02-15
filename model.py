from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss,confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
import pickle
import pandas as pd

marketing=pd.read_csv("data/marketing_new.csv",index_col=0)
X = marketing[['marketing_channel','subscribing_channel','variant']]
y = marketing['converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


categorical_features = X_train.select_dtypes(include=['object']).columns

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)])


def make_preds(classifier):
    pipe = Pipeline(steps=[('preprocessor', preprocessor),('classifier', classifier)])
    model=pipe.fit(X_train, y_train)
    print(classifier)
    print("model score: ",pipe.score(X_test, y_test))
    y_pred = pipe.predict(X_test)
    return model,y_pred

dt_model,dt_pred=make_preds(DecisionTreeClassifier(max_depth=4))

pickle.dump(dt_model,open('model.pkl','wb'))