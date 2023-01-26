import importlib
from .linear_regression import LinearRegression

class LinearRegression_sklearn(LinearRegression):
    """Wrapper for sklearn linear regression models 
    
    Wrapper for sklearn linear regression models available under 
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model 
    and
    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    
    Inherit base functions from models.linearregression.linear_regression.LinearRegression
    
    Parameters
    ----------    
    scalers : dict 
        dictionary with feature packages and respective scalers
    model : str, default='Ridge'
        name of sklearn model to use
    lib : str, default='sklearn'
    path : None or str, optional
        path to pretrained model
        default is None (model will be trained)
    dim : int or None, default=None
        dimension of features input, irrelevant for sklearn models
    config : dict, optional
        other configuration parameters for the sklearn linear regression models
        see sklearn's documentation for possible configurations
    
    Attributes
    ----------
    model : model-object
        sklearn model
    _scalers : dict
        dictionary with feature packages and respective scalers

    Notes
    -----
    define in config file under models via:    
    "models":[    
                {    
                    "lib": "sklearn",                    
                    "model": "Ridge",
                    "alpha":0.9
                    
                }
            ]      
    """
    def __init__(self, scalers, model='Ridge', lib='sklearn', path=None, dim=None, **config):
        if path:
            self.model = self._load_model(path)
        else:
            if model == 'MLPRegressor':
                self.model = self._import_dynamically('sklearn.neural_network', model)(**config)
            else:
                self.model = self._import_dynamically('sklearn.linear_model', model)(**config)
        self._scalers = scalers
    
    def _import_dynamically(self, module, model):
        """Import models dynamically 
        
        Parameters
        ----------        
        module : str
            name of sklearn module to import class form
        model : str
            name of model class
            
        Returns
        -------
        class
            class in sklearn module to import    
               
        Raises
        ------
        AttributeError
            if class not found
        """
        try:
            return getattr(importlib.import_module(module), model)
        except AttributeError as e:
            raise AttributeError("The chosen model {} does not exist in sklearn library. Make sure the model parameter doesn't have any typos and you have chosen the right library (lib parameter).").format(model) from e
    