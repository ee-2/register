from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from .linear_regression import LinearRegression

class LinearRegression_scorer(LinearRegression):
    """Linear Regression model for formality words scoring
    
    Inherit base functions from models.linearregression.linear_regression.LinearRegression
    
    Parameters
    ----------    
    path : None or str, optional
        path to pretrained formality scoring model
        default is None
    early_stopping : bool, default=True
        whether to use early stopping to terminate training when validation score is not improving
    max_iter : int, default=1000
        maximum number of iterations
    config : dict, optional
        other configuration parameters for the sklearn MLPRegressor
    
    Attributes
    ----------
    model : AdaBoostRegressor
        AdaBoostRegressor with MLPRegressor as base estimator
    """
    def __init__(self, path=None, early_stopping=True, max_iter=1000, **config):
        if path:
            self.model = self._load_model(path)
        else:
            self.model = AdaBoostRegressor(base_estimator=MLPRegressor(early_stopping=early_stopping,
                                                                       max_iter=max_iter,
                                                                       **config), 
                                           n_estimators=30, 
                                           learning_rate=1e-2)

    def dump(self, model_name, output_path):
        """ Dump model 
        
        Parameters
        ----------
        model_name : str
        output_path : str
        """
        self._dump_model(model_name, output_path)