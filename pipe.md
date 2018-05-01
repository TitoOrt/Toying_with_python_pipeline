
# Toying with SKLEARN Pipeline command

##### An example using pipline


```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'])
```


```python
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
```


```python
scaler = StandardScaler()
pca = PCA()
ridge = Ridge()
```


```python
X_train = scaler.fit_transform(X_train)
X_train = pca.fit_transform(X_train)
ridge.fit(X_train, y_train)
```




    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
       normalize=False, random_state=None, solver='auto', tol=0.001)




```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reduce_dim', PCA()),
        ('regressor', Ridge())
        ])
```


```python
pipe
```




    Pipeline(memory=None,
         steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)), ('regressor', Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
       normalize=False, random_state=None, solver='auto', tol=0.001))])




```python
pipe = pipe.fit(X_train, y_train)
print('Testing score: ', pipe.score(X_test, y_test))
```

    Testing score:  -17877.4266551
    


```python
ridge.score(X_test, y_test)
```




    -17877.426655125084



An explanation found in StackOverflow about the difference between ```fit()``` and ```fit_transform()``` methods: 

>To center the data (make it have zero mean and unit standard error), you subtract the mean and then divide the result by the standard deviation.

>$$x′=\frac{x−μ}{σ}$$

>You do that on the training set of data. But then you have to apply the same transformation to your testing set (e.g. in cross-validation), or to newly obtained examples before forecast. But you have to use the same two parameters $μ$ and $σ$ (values) that you used for centering the training set.

>Hence, every sklearn's transform's ```fit()``` just calculates the parameters (e.g. $μ$ and $σ$ in case of StandardScaler) and saves them as an internal objects state. Afterwards, you can call its ```transform()``` method to apply the transformation to a particular set of examples.

>Basically, ```fit_transform()``` joins these two steps and is used for the initial fitting of parameters on the training set $x$, but it also returns a transformed $x′$. Internally, it just calls first ```fit()``` and then ```transform()``` on the same data.

![image.png](attachment:image.png)


```python
import numpy as np
n_features_to_test = np.arange(1, 11)
alpha_to_test = 2.0**np.arange(-6, +6)params = {'reduce_dim__n_components': n_features_to_test,\
              'regressor__alpha': alpha_to_test}
```


```python
params = {'reduce_dim__n_components': n_features_to_test,\
              'regressor__alpha': alpha_to_test}
```


```python
from sklearn.model_selection import GridSearchCV
gridsearch = GridSearchCV(pipe, params, verbose=1).fit(X_train, y_train)
print('Final score is: ', gridsearch.score(X_test, y_test))
```

    Fitting 3 folds for each of 120 candidates, totalling 360 fits
    Final score is:  -12455.7192124
    

    [Parallel(n_jobs=1)]: Done 360 out of 360 | elapsed:    1.3s finished
    


```python
scalers_to_test = [StandardScaler(), RobustScaler(), QuantileTransformer()]
```


```python
params = {'scaler': scalers_to_test, 'reduce_dim__n_components': n_features_to_test, 'regressor__alpha': alpha_to_test}
```


```python
params = [
        {'scaler': scalers_to_test,
         'reduce_dim': [PCA()],
         'reduce_dim__n_components': n_features_to_test,\
         'regressor__alpha': alpha_to_test},

        {'scaler': scalers_to_test,
         'reduce_dim': [SelectKBest(f_regression)],
         'reduce_dim__k': n_features_to_test,\
         'regressor__alpha': alpha_to_test}
        ]
```


```python
gridsearch = GridSearchCV(pipe, params, verbose=1).fit(X_train, y_train)
```

    Fitting 3 folds for each of 720 candidates, totalling 2160 fits
    Final score is:  -16916.299943
    

    [Parallel(n_jobs=1)]: Done 2160 out of 2160 | elapsed:   25.7s finished
    


```python
print('Final score is: ', gridsearch.score(X_test, y_test))
```

    Final score is:  -16916.299943
    
