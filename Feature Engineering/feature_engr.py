import pandas as pd

df = pd.read_csv('/content/winequality-white.csv', sep=';')

from sklearn.preprocessing import StandardScaler


numerical_features = df.select_dtypes(include=['float64', 'int64'])


scaler = StandardScaler()


scaled_features = scaler.fit_transform(numerical_features)

scaled_df = pd.DataFrame(scaled_features, columns=numerical_features.columns)

final_df = pd.concat([scaled_df, df.select_dtypes(exclude=['float64', 'int64'])], axis=1)




final_df

from sklearn.preprocessing import MinMaxScaler

numerical_features = df.select_dtypes(include=['float64', 'int64'])


scaler = MinMaxScaler()


normalized_features = scaler.fit_transform(numerical_features)


normalized_df = pd.DataFrame(normalized_features, columns=numerical_features.columns)


final_df = pd.concat([normalized_df, df.select_dtypes(exclude=['float64', 'int64'])], axis=1)


final_df

from sklearn.ensemble import RandomForestRegressor


X = df.drop('pH', axis=1)
y = df['fixed acidity']


rf = RandomForestRegressor()


rf.fit(X, y)


feature_importances = rf.feature_importances_

top_feature_indices = feature_importances.argsort()[-5:][::-1]  # Select top 5 features


selected_features = X.columns[top_feature_indices]
print("Selected Features:")
print(selected_features)


interaction_terms = pd.DataFrame()
for i in range(len(df.columns)):
    for j in range(i+1, len(df.columns)):
        colname = df.columns[i] + "_x_" + df.columns[j]
        interaction_terms[colname] = df.iloc[:, i] * df.iloc[:, j]

wine_data_interacted = pd.concat([df, interaction_terms], axis=1)

wine_data_interacted.head()


