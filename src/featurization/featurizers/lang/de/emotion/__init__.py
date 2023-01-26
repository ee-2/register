class EmotionFeaturizer():
    """Get emotion features for German
    
    Combine featurization of language independent emotion feature package with German sentiment feature package

    Parameters
    ----------
    name : str
        name of the feature package
    level : {'document','token','sentence'}
        level to calculate formality on, default is 'document'
    path : str or dict, optional
        path to memolon lexicon
        default loads the lexicon for a specific language (following its ISO code) from folder lang_data/emotion
        for multiple languages path is a dictionary with the languages' ISO code as keys and the respective paths as values
        if only one language is used path can be a string containing the path
    features: list, default=['valence','arousal','dominance','joy','anger','sadness','fear','disgust','polarity']
        emotion features
    vectorizer : dict, default={'name':'TfidfVectorizer',"use_idf":False, "norm":"l1"} 
        vectorizer to vectorize features with,
        for level 'document' no vectorizer is taken
        choose from CountVectorizer(https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
        or TfidfVectorizer (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) from sklearn
    scaler : dict or bool or None, optional
        scaler to scale data features with
        set to 'false' if you do not want to scale the features,
        choose from scalers provided by sklearn (e.g. StandardScaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) 
        or MinMaxScaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler))
        defaults to None, which takes StandardScaler ({'name':'StandardScaler'}) for level 'token' or 'sentence' and for level 'document' MinMaxScaler ({'name':'MinMaxScaler'})

    Attributes
    ----------
    config : dict
        the configurations for the feature package 
    _modules : list
        list of combined modules (language independent feature package and German sentiment feature package)

    Notes
    -----
    define in config file under feature_packages via:
        {
            "feature_package": "emotion"
        }   
    """
    def __init__(self, nlp, name, path='', features=['valence','arousal','dominance','joy','anger','sadness','fear','disgust', 'polarity'], level='document',
                 vectorizer={'name':'TfidfVectorizer',"use_idf":False, "norm":"l1"}, scaler=None):
        self.config = {'name':name,
                       'features':features,
                       'level':level}
        if level == 'document':
            self.config['scaler'] = {'name':'MinMaxScaler'} if scaler == None else scaler
        else:
            self.config['scaler'] = {'name':'StandardScaler'} if scaler == None else scaler
            self.config['vectorizer'] = vectorizer
        self._modules = []
        emo_features = [feature for feature in features if feature != 'polarity']
        if emo_features:
            from ....emotion import EmotionFeaturizer
            self._modules.append(EmotionFeaturizer(nlp, 'de', name, path=path, features=emo_features, level=level))
        if 'polarity' in features:
            from .sentiment import SentimentFeaturizer
            self._modules.append(SentimentFeaturizer(features=['polarity'], level=level))

    def featurize_doc(self, doc):
        """Get features for doc 

        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        dict
            dictionary with feature names as keys, 
            values are either list of emotion scores (for level 'token' or 'sentence') or emotion score itself (for level 'document') for the specific feature
        """ 
        data = {}
        for module in self._modules:
            data.update(module.featurize_doc(doc))
        return {feature:data[feature] for feature in self.config['features']}
    