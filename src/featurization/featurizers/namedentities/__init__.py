class NamedEntityFeaturizer():
    """Get named entities
    
    Get specified named entities on span level (one entity may contain multiple tokens)
    
    Parameters
    ----------
    name : str
        name of the feature package
    entities : list, optional
        entities to take into account (e.g. ['ORG','LOC']), default all entities specified for language
        (called entities and not features, because consistency between different languages is not given, feature package returns list not metrics)   
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
    
    Notes
    -----
    define in config file under feature_packages via:
            {
                "feature_package": "named_entities",
                "entities": ["ORG"],
                "vectorizer":    {
                                    "name": "TfidfVectorizer"
                                 },
                "scaler":        {
                                    "name": "Normalizer"
                                 }
            }
    """   
    def __init__(self, nlp, name, entities=[], 
                 vectorizer={"name":"CountVectorizer", "max_features":1000},
                 scaler={'name':'StandardScaler'}):
        
        self.config={'name':name,
                     'entities':entities if entities else nlp.meta['labels']['ner'],
                     'vectorizer':vectorizer,
                     'scaler':scaler}
        self._name = name
        
        
    def featurize_doc(self, doc):
        """Get features for doc 
        
        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        dict
            dictionary with feature package name as key and list of entities as value
        """
        return {self._name: [ent.label_ for ent in doc.ents if ent.label_ in self.config['entities']]} 
         
        