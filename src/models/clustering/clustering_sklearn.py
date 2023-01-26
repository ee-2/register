import importlib
from .clustering import Clustering

class Clustering_sklearn(Clustering):
    """Wrapper for sklearn clustering models
    
    Wrapper for sklearn clustering models available under 
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster

    Inherit base functions from models.clustering.Clustering
    
    Parameters
    ----------    
    scalers : dict 
        dictionary with feature packages and respective scalers
    model : str, default='AgglomerativeClustering'
        name of sklearn model to use
    lib : str, default='sklearn'
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
                    "model": "KMeans",
                    "n_clusters":3
                    
                }
            ]      
    """    
    def __init__(self, scalers, model='AgglomerativeClustering', lib='sklearn', **config):
        self.model = self._import_dynamically('sklearn.cluster', model)(**config)
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