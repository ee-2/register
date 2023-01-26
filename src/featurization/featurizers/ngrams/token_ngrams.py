class TokenNgramsFeaturizer():
    """Get ngrams of attributes from spacy token
    
    Default run: return unigrams of lemmas
    
    Parameters
    ----------
    name : str
        name of the feature package
        if you want to stack multiple token_ngrams feature packages, set an unique name to avoid conflicts
    n : int, default=1
        choose the n for the n-grams, i.e. 2 for bigrams
    spacy_attrib : str, default="lemma_"
        spacy attribute to get ngrams from, i.e. text, lemma, POS-tag etc.
        see spaCy's Token docu (https://spacy.io/api/token) for possible attributes
    exclude_spacy_attribs : list, default=[]
        define list with spacy attributes which should not be indcluded, e.g. punctuation or stop words
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
    _exclude_spacy_attribs : list
        list with spacy attributes which should not be indcluded
    
    Notes
    -----
    define in config file under feature_packages via:
            {
                "feature_package": "token_ngrams",
                "name": "token_bi",
                "n":2, 
                "exclude_spacy_attribs":["is_stop"],
                "spacy_attrib":"text",
                "vectorizer":    {
                                    "name":"CountVectorizer",
                                    "binary": true
                                 },
                "scaler": false
            },
            {
                "feature_package": "token_ngrams",
                "name": "lemma_uni"         
            }
    """
    def __init__(self, name, n=1, spacy_attrib='lemma_', exclude_spacy_attribs=[], 
                 vectorizer={"name":"CountVectorizer", "max_features":1000},
                 scaler={'name':'StandardScaler'}):
        self.config={'name':name,
                     'n':n,
                     'spacy_attrib':spacy_attrib,
                     'exclude_spacy_attribs':exclude_spacy_attribs,
                     'vectorizer':vectorizer,
                     'scaler':scaler}
        self._name = name   
        self._n = n
        self._spacy_attrib = spacy_attrib
        self._exclude_spacy_attribs = exclude_spacy_attribs     
        
    def featurize_doc(self, doc):
        """Get features for doc 
        
        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        dict
            dictionary with feature package name as key and list of ngrams as value, e.g., {"token_bi":["bigram1", "bigram2"]}
        """
        return {self._name:self._featurize_doc(doc)}
    
    def _featurize_doc(self, doc):
        """Get ngrams 
        
        Get ngrams list of all spacy_attribs (only one possible) provided for spacy token, 
        tokens with specific attribs set in exclude_spacy_attribs will be excluded
        
        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        list
        """
        if self._n == 1:
            return [getattr(token, self._spacy_attrib) for token in doc if not any(getattr(token, excl_attr) for excl_attr in self._exclude_spacy_attribs)]
        else:
            return [' '.join([getattr(token, self._spacy_attrib) for token in ngram]) for ngram in [doc[i:i+self._n] for i in range(len(doc)-self._n+1)] 
                    if not any(getattr(token, excl_attr) for token in ngram for excl_attr in self._exclude_spacy_attribs)]       
         
        