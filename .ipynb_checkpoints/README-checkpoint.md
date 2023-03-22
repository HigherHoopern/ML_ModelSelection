# ML_ModelSelection

This package is to facilitate model selection in Machine Learning. It is a common issue that ML practioners do not know which model to select prior to optimization, since tuning hypermeters is time consuming and computional. To make life easier, this package allows users  to train a couple of machine leanring models using their default hypermeteres, and compare their performance to determine which model to select. 

# Usage

`pip install mlms`

`from MLMS import ModelSelection as MS`

`performance, models = MS.Select_Classifier('accuracy', 10, X_train, X_test, y_train, y_test)`


Then instantiate and use it like this:

from ModelSelection(mode='classification',scoring='accuracy',K_folds=5)

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
