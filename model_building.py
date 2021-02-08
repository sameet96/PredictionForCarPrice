import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle
from pandas.plotting import scatter_matrix

df = pd.read_csv('cleaned_data.csv')

print(df['make'].value_counts().count())
df['make'].replace("MERCEDES-BENZ", "MERCEDES_BENZ", inplace=True)

sns.pairplot(df)
plt.show()

corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
# plot heat map
g = sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()

a = df.describe()

print(df['car_type'].value_counts())
print(df['exterior_color'].value_counts())

attributes = ["year", "model", "exterior_color", "interior_color", "number_of_keys", "price", "miles", "make",
              "highway_mpg", "city_mpg", "car_type", "number_of_gears", "type_of_transmission", "number_of_doors"]
scatter_matrix(df[attributes], figsize=(12, 8))
plt.savefig('matrix.png')
plt.show()

df.drop(columns=["interior_color", "model", "number_of_keys", "number_of_doors"], inplace=True)
df['number_of_gears'] = df['number_of_gears'].replace({"CVT": "0"})

df['price'] = df['price'].astype(int)
# creating dummies

df = pd.get_dummies(df, columns=['exterior_color', 'type_of_transmission', 'car_type', 'make'],
                    prefix=['exterior_color', 'type_of_transmission', 'car_type', 'make'])
df

df['miles'] = df['miles'].str.replace('|', "")
df['miles'] = df['miles'].str.lstrip()
df[['price', 'year', 'miles', 'highway_mpg', 'city_mpg', 'number_of_gears']] = df[
    ['price', 'year', 'miles', 'highway_mpg', 'city_mpg', 'number_of_gears']].astype(int)

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# important feature


model = ExtraTreesRegressor()
model.fit(X, y)

# plotting 7 most important features
important_feature = model.feature_importances_
feat_importance = pd.Series(important_feature, index=X.columns)
feat_importance.nlargest(8).plot(kind='barh')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=38)

# HyperParameter
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
#
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf}
#
# rf_random_forest = RandomForestRegressor()
# rf_random = RandomizedSearchCV(estimator=rf_random_forest, param_distributions=random_grid,
#                                scoring='neg_mean_squared_error', n_iter=10, cv=15, verbose=2, random_state=38, n_jobs=1)
rf_random = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=30, max_features='sqrt', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=1000, n_jobs=None, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)
rf_random.fit(X_train, y_train)

prediction_for_random_forest = rf_random.predict(X_test)



plt.scatter(y_test, prediction_for_random_forest)
plt.show()

from sklearn.model_selection import cross_val_score

scoreForRandomForest = cross_val_score(rf_random, X_test, prediction_for_random_forest, cv=10)
print("Score for random forest " + str(scoreForRandomForest.mean))
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, prediction_for_random_forest))
print('MSE:', metrics.mean_squared_error(y_test, prediction_for_random_forest))
rmse_for_random_forest = np.sqrt(metrics.mean_squared_error(y_test, prediction_for_random_forest))
print('RMSE:', rmse_for_random_forest)


####### Best param for randomforest


#
# (bootstrap=True, ccp_alpha=0.0, criterion='mse',
#                       max_depth=30, max_features='sqrt', max_leaf_nodes=None,
#                       max_samples=None, min_impurity_decrease=0.0,
#                       min_impurity_split=None, min_samples_leaf=1,
#                       min_samples_split=2, min_weight_fraction_leaf=0.0,
#                       n_estimators=1000, n_jobs=None, oob_score=False,
#                       random_state=None, verbose=0, warm_start=False)

# Hyper Parameter Optimization
#
# params = {
#     "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
#     "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
#     "min_child_weight": [1, 3, 5, 7],
#     "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
#     "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
#
# }
#
#
# def timer(start_time=None):
#     if not start_time:
#         start_time = datetime.now()
#         return start_time
#     elif start_time:
#         thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
#         tmin, tsec = divmod(temp_sec, 60)
#         print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
#
#
# from sklearn.model_selection import RandomizedSearchCV
# import xgboost
#
# classifier = xgboost.XGBClassifier()
#
# random_search_for_xgboost_parameters = RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, n_jobs=-1,
#                                                           cv=5, verbose=3)
# from datetime import datetime
#
# # Here we go
# start_time = timer(None)  # timing starts from this point for "start_time" variable
# random_search_for_xgboost_parameters.fit(X_train, y_train)
# timer(start_time)  # timing ends here for "start_time" variable
#
# xgboost_regressor = xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#                                          colsample_bynode=1, colsample_bytree=0.7, gamma=0.4, gpu_id=-1,
#                                          importance_type='gain', interaction_constraints='',
#                                          learning_rate=0.2, max_delta_step=0, max_depth=12,
#                                          min_child_weight=1, missing=None, monotone_constraints='()',
#                                          n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
#                                          reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#                                          tree_method='exact', validate_parameters=1, verbosity=None)
#
# predForXgboost = random_search_for_xgboost_parameters.predict(X_test)
#
# # sns.displot(y_test - predForXgboost)
# # plt.show()
# from sklearn import metrics
#
# print('MAE:', metrics.mean_absolute_error(y_test, predForXgboost))
# print('MSE:', metrics.mean_squared_error(y_test, predForXgboost))
# rmse_for_xgboost = np.sqrt(metrics.mean_squared_error(y_test, predForXgboost))
# print('RMSE:', rmse_for_xgboost)
#
# # regressor_xgboost = xgboost.best_for_xgboost
# from sklearn.model_selection import cross_val_score
#
# scoreForXgboost = cross_val_score(classifier, X_test, predForXgboost, cv=10)
# print("Score for Xgboost" + str(scoreForXgboost.mean))



# selecting the best model to deploy

#
# if rmse_for_random_forest < rmse_for_xgboost:
#     file_name = 'random_forest_regression.pkl'
#     file = open(file_name, 'wb')
#     pickle.dump(rf_random, file)
# else:
#     file_name = 'xgboost_regression.pkl'
#     file = open(file_name, 'wb')
#     pickle.dump(random_search_for_xgboost_parameters, file)

# import pickle
#
# file = open('random_forest_regression.pkl', 'wb')
#
# pickle.dump(rf_random, file)
