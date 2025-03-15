from sklearn.pipeline import Pipeline
##feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
steps=[("standard_scaler",StandardScaler()),
      ("classifier",LogisticRegression())]
steps
pipe=Pipeline(steps)
##visualize Pipeline
from sklearn import set_config
set_config(display="diagram")
pipe
##creating a dataset
from sklearn.datasets import make_classification
X,y=make_classification(n_samples=1000)
X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
X_train
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
y_pred

# Displaying a pipeline with standard scaler, dimesnionality reduction and then estimator
from sklearn.decomposition import PCA
from sklearn.svm import SVC
steps=[("scaling",StandardScaler()),
      ("PCA",PCA(n_components=3)),
      ("SVC",SVC())]
pipe2=Pipeline(steps)
pipe2.fit(X_train,y_train)
pipe2.predict(X_test)

# Column Transformer
from sklearn.impute import SimpleImputer
## numerical processing pipeline
import numpy as np
numeric_processor=Pipeline(
    steps=[("imputation_mean",SimpleImputer(missing_values=np.nan,strategy="mean")),
          ("scaler",StandardScaler())]

)
numeric_processor
##categorical procesing pipeline
from sklearn.preprocessing import OneHotEncoder
categorical_processor=Pipeline(
    steps=[("imputation_consatnt",SimpleImputer(fill_value="missing",strategy="constant")),
          ("onehot",OneHotEncoder(handle_unknown="ignore"))]

)
categorical_processor
## combine processing technqiues
from sklearn.compose import ColumnTransformer
preprocessor=ColumnTransformer(
    [("categorical",categorical_processor,["gender","City"]),
    ("numerical",numeric_processor,["age","height"])]


)

preprocessor
from sklearn.pipeline import make_pipeline
pipe=make_pipeline(preprocessor,LogisticRegression())
pipe
