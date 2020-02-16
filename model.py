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
X = marketing[['marketing_channel','subscribing_channel','age_group']]
y = marketing['converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


categorical_features = X_train.select_dtypes(include=['object']).columns

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)])


def fit_model(classifier):
    pipe = Pipeline(steps=[('preprocessor', preprocessor),('classifier', classifier)])
    model=pipe.fit(X_train, y_train)
    return pipe,model

pipeline,model=fit_model(DecisionTreeClassifier(max_depth=4))
column_names = model.named_steps['preprocessor'].transformers_[0][1]\
    .named_steps['onehot'].get_feature_names(categorical_features)




pickle.dump(model,open('model.pkl','wb'))
pickle.dump(column_names,open('model_columns.pkl','wb'))