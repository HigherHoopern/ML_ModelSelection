# Machine Learning Model Selection

This package aims to facilitate model selection in Machine Learning. It is a common issue that ML practitioners often struggle to decide on the most appropriate model prior to optimization, as tuning hyperparameters can be time-consuming and computationally demanding. To simplify the process, this package enables users to train several machine learning models using their default hyperparameters and compare their performance, helping them determine the most suitable model to select.

# Usage

`pip install mlms -U`

`https://pypi.org/project/mlms/`

Then instantiate and use it like this:

`from mlms.ModelSelection import Select_Regressor, Select_Classifier`

`df_performance, fitted_classifiers = Select_Classifier('accuracy', 10, X_train, X_test, y_train, y_test)`

`df_performance, fitted_regressors = Select_Classifier('neg_mean_squared_erro', 10, X_train, X_test, y_train, y_test)`

For classifiers, the performance can set as `accuracy` , `f1_score` , `precision`, `recall`, `roc_auc` and so on. Available classifiers are below

* `('LGR', LogisticRegression(n_jobs=-1))`,
* `('AB', AdaBoostClassifier())`,
* `('CART', DecisionTreeClassifier())`,
* `('GBC', GradientBoostingClassifier())`,
* `('XGBC', XGBClassifier())`,
* `('RFC', RandomForestClassifier())`,
* `('ETC', ExtraTreeClassifier())`,
* `('KNN', KNeighborsClassifier(n_jobs=-1))`,
* `('NB', GaussianNB())`,
* `('SVC', SVC())`,
* `('MLP', MLPClassifier()),`
* `('SGDC', SGDClassifier(n_jobs=-1)),`
* `('GPC', GaussianProcessClassifier(n_jobs=-1)),`
* `('PAC', PassiveAggressiveClassifier(n_jobs=-1))`

(The charts is an classifier selection example using Iris dataset)

![1679444303986](image/README/1679444303986.png)

![1679443565646](image/README/1679443565646.png)

![1679443664816](image/README/1679443664816.png)

For regressors, the performance can set as `r2_score`, `neg_mean_squared_error` and so on. Available regressors are below:

- `('KNN', KNeighborsRegressor())`,
- `('CART', DecisionTreeRegressor())`,
- `('SVR', SVR()),`
- `('MLP', MLPRegressor())`,
- `('ABR', AdaBoostRegressor())`,
- `('GBR', GradientBoostingRegressor())`,
- `('XGB', XGBRegressor())`,
- `('RFR', RandomForestRegressor())`,
- `('ETR', ExtraTreesRegressor())`

![1679487197758](image/README/1679487197758.png)

[GitHub](https://github.com/HigherHoopern/ML_ModelSelection)
