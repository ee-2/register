import importlib
from .classification import Classification

class Classification_sklearn(Classification):
    """Wrapper for sklearn classification models 
    
    Wrapper for sklearn classifiers:
    Logistic Regression, 
    SVMs, 
    Linear Classifiers with Stochastic Gradient Descent Training, 
    Decision Tree, 
    Nearest Neighbors, 
    Naive Bayes and 
    Multi-layer Perceptron
    
    Inherit base functions from models.classification.Classification
    
    Parameters
    ----------    
    scalers : dict 
        dictionary with feature packages and respective scalers
    model : str, default='SGDClassifier'
        name of sklearn model to use
    lib : str, default='sklearn'
    path : None or str, optional
        path to pretrained model
        default is None (model will be trained)
    nr_labels : int or None
        number of possible class labels
        (not relevant for sklearn models yet,
        can be extended for different multiclass strategies later on)
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
                    "model": "LogisticRegression",
                    "penalty":"l2"
                    
                }
            ]      
    """
    def __init__(self, scalers, model='SGDClassifier', lib='sklearn', path=None, dim=None, nr_labels=None, **config):
        self._scalers = scalers
        if path:
            self.model = self._load_model(path)
        else:
            if model in ['LogisticRegression','SGDClassifier']:
                self.model = self._import_dynamically('sklearn.linear_model', model)(**config)  
            elif model == 'DecisionTreeClassifier':
                self.model = self._import_dynamically('sklearn.tree', model)(**config)
            elif model in ['SVC','NuSVC','LinearSVC']:
                self.model = self._import_dynamically('sklearn.svm', model)(**config)
            elif model == 'KNeighborsClassifier':
                self.model = self._import_dynamically('sklearn.neighbors', model)(**config)
            elif model == 'GaussianNB':
                self.model = self._import_dynamically('sklearn.naive_bayes', model)(**config)
            elif model == 'MLPClassifier': 
                self.model = self._import_dynamically('sklearn.neural_network', model)(**config)

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