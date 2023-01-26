import numpy as np
import importlib
from sklearn.feature_extraction import DictVectorizer

class EmbeddingsVectorizer():
    """Vectorize embeddings 
    """
    def vectorize(self, x):
        """Stack embeddings vertically
        
        Parameters
        ----------
        x : list of ndarray or tensors
        
        Returns
        -------
        ndarray
        """
        try:
            return np.vstack(x)
        except TypeError:
            return np.vstack([emb.cpu().numpy() for emb in x]) 
        


class VectorizerFrequency_sklearn():
    """Wrapper for sklearn's vectorizers
    
    Vectorizers provided by sklearn in sklearn.feature_extraction.text determined in feature package configs
    Uses custom analyzer
    
    Parameters
    ----------    
    x : None or list of lists, optional
        list with list of n-grams per doc
        default is None
    vectorizer : None or CountVectorizer or TfidfVectorizer
        pretrained vectorizer
    name : {'CountVectorizer','TfidfVectorizer}
        name of vectorizer to import
    config : dict, optional
        other configuration parameters for the sklearn vectorizers
    
    Attributes
    ----------
    vectorizer : CountVectorizer or TfidfVectorizer
    
    Notes
    -----    
    example config for relative frequencies normalized by the document length
    
         "vectorizer": {"name": "TfidfVectorizer",
                        "use_idf": false,
                        "norm": "l1"
                        }
    """
    def __init__(self, x=None, vectorizer=None, name='CountVectorizer', **config):
        if vectorizer:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = self._import_dynamically(name)(analyzer=self._analyze, **config)
            self.vectorizer.fit(x)
    
    def vectorize(self, x):
        """Vectorize x and transforms x to array (from sparse matrix)
        
        Parameters
        ----------        
        x : None or list of lists, optional
            list with list of n-grams per doc
            default is None
        
        Returns
        -------
        ndarray
        """
        return self.vectorizer.transform(x).toarray()

    def _import_dynamically(self, name):
        """Import vectorizer dynamically 
        
        Parameters
        ----------        
        name : str
            name of vectorizer to import

        Returns
        -------
        class
            sklearn Vectorizer
        
        Raises
        ------
        AttributeError
            if vectorizer not found
        """
        try:
            return getattr(importlib.import_module("sklearn.feature_extraction.text"), name)
        except AttributeError as e:
            raise AttributeError("The chosen vectorizer does not exist in sklearn library. Make sure the name parameter doesn't have any typos.") from e
    
    def _analyze(self, x):
        """Helper to prevent sklearn from tokenizing 
        Parameters
        ----------        
        x : list of lists
        
        Returns
        -------
        x : list of lists
        """
        return x 
        

class VectorizerDict_sklearn():
    """Wrapper for sklearn's dict vectorizer 
    
    Default if no vectorizer is specified for feature package
    
    Parameters
    ----------    
    x : None or list of lists, optional
        list with list of n-grams per doc
        default is None
    vectorizer : None or CountVectorizer or TfidfVectorizer
        pretrained vectorizer
    sparse : bool, default=False
        whether transform should produce scipy.sparse matrices
    config : dict, optional
        other configuration parameters for the sklearn vectorizers
    
    Attributes
    ----------
    vectorizer : DictVectorizer
    """
    def __init__(self, x=None, vectorizer=None, sparse=False, **config):
        if vectorizer:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = DictVectorizer(sparse=sparse, **config)
            self.vectorizer.fit(x)
    
    def vectorize(self,x):
        """Vectorize x
        
        Parameters
        ----------        
        x : None or list of lists, optional
            list with list of n-grams per doc
            default is None
        
        Returns
        -------
        ndarray
        """
        return self.vectorizer.transform(x)

"""
define your own vectorization class here (see 3 classes above for examples)
- with function vectorize which takes x and returns it in a vectorized format as a 2-dimensional numpy array
- if vectorizer needs to be reused when loading a pretrained model, define a vectorizer attribute which contains the vectorizer model (either loaded from pretrained vectorizer or instantiated) 
"""