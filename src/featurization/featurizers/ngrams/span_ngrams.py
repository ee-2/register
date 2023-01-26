class SpanNgramsFeaturizer():
    """Get ngrams of attributes from spacy spans
    
    Default run: returns unigrams of entities

    Parameters
    ----------
    name : str
        name of the feature package
        if you want to stack multiple span_ngrams feature packages, set an unique name to avoid conflicts
    n : int, default=1
        choose the n for the n-grams, i.e. 2 for bigrams
    spacy_attrib : str, default="ents"
        spacy attribute to get ngrams from, i.e. ents or noun_chunks
        see spaCy's Token docu (https://spacy.io/api/token) for possible attributes
    vectorizer : dict, default={"name":"CountVectorizer", "max_features":1000} 
        vectorizer to vectorize features with
        choose from CountVectorizer(https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
        or TfidfVectorizer (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) from sklearn
    scaler : dict or bool, default={"name": "StandardScaler}
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
    _spacy_attrib : str
        spacy attribute to get ngrams from
    
    Notes
    -----    
    define in config file under feature_packages via:
        {
            "feature_package":"span_ngrams",
            "name": "named_entities"
        }
    """

    def __init__(self, name, n=1, spacy_attrib='ents', 
                 vectorizer={"name":"CountVectorizer", "max_features":1000},
                scaler={'name':'StandardScaler'}):
        self.config={'name':name,
                     'n':n,
                     'spacy_attrib':spacy_attrib,
                     'vectorizer':vectorizer,
                     'scaler':scaler}
        self._name = name   
        self._n = n 
        self._spacy_attrib = spacy_attrib
        
    def featurize_doc(self, doc):
        """Get features for doc 
        
        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        dict
            dictionary with feature package name as key and list of ngrams as value, e.g., {"named_entity":["PERSON", "LOCATION"]}
        """
        return {self._name:self._featurize_doc(doc)}
    
    def _featurize_doc(self, doc):
        """Get ngrams 
        
        Get ngrams list of of spans provided for spacy doc
        
        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        list
        """
        if self._n == 1:
            return [span.label_ for span in getattr(doc, self._spacy_attrib)]
        else:
            ngrams = [span.label_ for span in getattr(doc, self._spacy_attrib)]
            return [' '.join(ngrams[i:i+self._n]) for i in range(len(ngrams)-self._n+1)]        
