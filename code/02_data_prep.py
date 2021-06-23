import pandas as pd

from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv('./data/recruitment_details.csv')

df = df.drop(['sl_no', 'salary'], axis=1)

print(f"The dataset is composed of {df.iloc[:, :-1].shape[1]} features")

df['status'] = df['status'].map({'Placed': 1, 'Not Placed': 0})

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(df)
data_categorical = df[categorical_columns]

print(f"The categorical sub-ataset is composed of {data_categorical.shape[1]} features")

specialisation_column = data_categorical[["specialisation"]]

encoder = OneHotEncoder(sparse=False)
specialisation_encoded = encoder.fit_transform(specialisation_column)

feature_names = encoder.get_feature_names(input_features=["specialisation"])
specialisation_encoded = pd.DataFrame(specialisation_encoded, columns=feature_names)

data_encoded = encoder.fit_transform(data_categorical)

print(f"The encoded dataset contains {data_encoded.shape[1]} features")

columns_encoded = encoder.get_feature_names(data_categorical.columns)
encoded_df = pd.DataFrame(data_encoded, columns=columns_encoded)

new_df = pd.concat([df.drop(categorical_columns, axis=1), encoded_df], axis=1)
new_df = pd.concat([new_df.drop('status', axis=1), new_df[['status']]], axis=1)

new_df.to_csv('./data/prepared_data.csv')