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


# Apply one-hot encoding to categorical variables
df_encoded = pd.get_dummies(df)

df_encoded.head()

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column contains categorical data
        df[column] = label_encoder.fit_transform(df[column])

df.head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(wine_data_interacted.drop(columns=['pH']), wine_data_interacted['pH'], test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor()

# Fit the model on the training data
rf_regressor.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


import matplotlib.pyplot as plt

# Plot predicted vs actual values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual pH')
plt.ylabel('Predicted pH')
plt.title('Regression Results: Actual vs Predicted pH')
plt.show()


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor()

cv_scores = cross_val_score(rf_regressor, X, y, cv=5, scoring='neg_mean_squared_error')

cv_scores = -cv_scores

print("Cross-Validation Scores:")
print(cv_scores)

mean_cv_score = cv_scores.mean()
std_cv_score = cv_scores.std()

print("Mean Cross-Validation Score:", mean_cv_score)
print("Standard Deviation of Cross-Validation Scores:", std_cv_score)


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_regressor = RandomForestRegressor()

grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

grid_search.fit(X, y)


best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

best_score = grid_search.best_score_
print("Best Mean Cross-Validation Score:", -best_score)








