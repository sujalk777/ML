from sklearn.pipeline import Pipeline
##feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
steps=[("standard_scaler",StandardScaler()),
      ("classifier",LogisticRegression())]
