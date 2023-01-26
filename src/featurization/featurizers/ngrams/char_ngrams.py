class CharNgramsFeaturizer():
    """Gets ngrams of chars
    
    Default run: returns 3-grams of chars

    Parameters
    ----------
    name : str
        name of the feature package
        if you want to stack multiple character_ngrams feature packages, set an unique name to avoid conflicts
    n : int, default=3
        choose the n for the n-grams, i.e. 2 for bigrams
    cross_token_boundaries : bool, default=True
        whether to generate n-grams crossing token boundaries, if false tokens will be padded with whitespace to track start and end
    vectorizer : dict, default={"name":"CountVectorizer", "max_features":1000} 
        vectorizer to vectorize features with
        choose from CountVectorizer(https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
        or TfidfVectorizer (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) from sklearn
    scaler : dict or bool default={"name": "StandardScaler}
        scaler to scale features with
        set to 'false' if you do not want to scale the features
        choose from scalers provided by sklearn (e.g. StandardScaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) 
        or MinMaxScaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler))

    Attributes
    ----------
    config : dict
        the configurations for the feature package 
    _name : str
        name of the feature package
    _n : int
        the n for the n-grams
    _cross_token_boundaries : bool
        whether to generate n-grams crossing token boundaries
    
    Notes
    -----
    define in config file under feature_packages via:
            {
                "feature_package": "character_ngrams",
                "cross_token_boundaries":false
            }
    """
    def __init__(self, name, n=3, cross_token_boundaries=True, 
                 vectorizer={"name":"CountVectorizer", "max_features":1000},
                 scaler={'name':'StandardScaler'}):
        self.config={'name':name,
                     'n':n,
                     'cross_token_boundaries':cross_token_boundaries,
                     'vectorizer':vectorizer,
                     'scaler':scaler}
        self._name = name   
        self._n = n
        self._cross_token_boundaries = cross_token_boundaries 
        
    def featurize_doc(self, doc):
        """Get features for doc 
        
        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        dict
            dictionary with feature package name as key and list of ngrams as value, e.g., {"char_tri":["fun", "unk"]}
        """
        return {self._name:self._featurize_doc(doc)}
        
    def _featurize_doc(self, doc):
        """Get ngrams 
        
        Get ngrams list of characters, 
        if cross_token_boundaries is true generate n-grams crossing token boundaries, if false tokens will be padded with white space
        
        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        list
        """
        if self._n == 1:
            return [char for token in doc for char in token.text]
        else:
            if self._cross_token_boundaries:
                text = ' '+doc.text_with_ws+' '
                return [text[i:i+self._n] for i in range(len(text)-self._n+1)]  
            else:
                ngrams = [' '+token.text+' ' for token in doc]
                return [token[i:i+self._n] for token in ngrams for i in range(len(token)-self._n+1)]
        