import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


file_path = './data/prepared_data.csv'

# Loading preprocessed data
recruitment_df = pd.read_csv(file_path)

# Extract X and y variables and values
target_name = "status"
X = recruitment_df.drop([target_name], axis = 1)
y = recruitment_df[target_name]

# Scaling
z = X.values
min_max_scaler = MinMaxScaler()
z_scaled = min_max_scaler.fit_transform(z)
X = pd.DataFrame(z_scaled)

# Split dataset into train and test sets
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 823)

# KNN

modelKNN = KNeighborsClassifier(n_neighbors = 12, weights='distance')
modelKNN.fit(X_train, y_train)
predictionsKNN = modelKNN.predict(X_test)
accuracyKNN = metrics.accuracy_score(y_test, predictionsKNN)  
print(accuracyKNN)

# Logistic Regression
modelLogReg = LogisticRegression()
modelLogReg.fit(X_train, y_train)
predictionsLogReg = modelLogReg.predict(X_test)
accuracyLogReg = modelLogReg.score(X_test, y_test)

print(accuracyLogReg)


# Metrics json file
with open("./data/metrics.json", 'w') as outputfile:
        json.dump(
        	{ 
        	  "accuracy_KNN"                   : accuracyKNN,
        	  "accuracy_logistic-regression"   : accuracyLogReg,
        	}, 
        	outputfile
        )