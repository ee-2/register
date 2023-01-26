from spacy.morphology import Morphology

class MorphologyFeaturizer():
    """Get morphological features
    
    Morphological attributes for spacy token based on: https://universaldependencies.org/format.html#morphological-annotation

    Parameters
    ----------
    name : str
        name of the feature package
    morph_tags : list, optional
        morphological tags to take into account, default takes all morphological tags (defined under https://universaldependencies.org/u/feat/index.html):
        ['Abbr','AdpType','AdvType','Animacy','Aspect','Case','Clusivity','ConjType',
         'Definite','Degree','Deixis','DeixisRef','Echo','Evident','Foreign','Gender',
         'Gender[dat]','Gender[erg]','Gender[obj]','Gender[psor]','Gender[subj]','Hyph',
         'Mood','NameType','NounClass','NounType','NumForm','NumType','NumValue',
         'Number','Number[abs]','Number[dat]','Number[erg]','Number[obj]','Number[psed]',
         'Number[psor]','Number[subj]','PartType','Person','Person[abs]','Person[dat]',
         'Person[erg]','Person[obj]','Person[psor]','Person[subj]','Polarity','Polite',
         'Polite[abs]','Polite[dat]','Polite[erg]','Poss','PrepCase','PronType','PunctSide',
         'PunctType','Reflex','Style','Subcat','Tense','Typo','VerbForm','VerbType','Voice']
         (called morph_tags and not features, because consistency between different languages is not given, feature package returns list not metrics)
        Availability depends on language. For German, e.g., only the following tags are available:
        ['Definite', 'Tense', 'Poss', 'Person', 'VerbForm', 'Number', 'Foreign', 'Degree', 'Mood', 'Gender', 'Reflex', 'Case', 'PronType'] and POS (general pos tag)
    vectorizer : dict, default={"name":"CountVectorizer", "max_features":1000} 
        vectorizer to vectorize features with
        choose from CountVectorizer(https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
        or TfidfVectorizer (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) from sklearn
    scaler : dict, default={"name": "StandardScaler}
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
                "feature_package": "morphology",
                "name": "CNG",
                "morph_tags": ["Case","Number","Gender"],
                "vectorizer":    {
                                    "name": "TfidfVectorizer"
                                 },
                "scaler":        {
                                    "name": "Normalizer"
                                 }
            },
            {
                "feature_package": "morphology",
                "name": "Tense"
                "features": ["Person"]            
            }
    """    
    def __init__(self, 
                 name, 
                 morph_tags=['Abbr','AdpType','AdvType','Animacy','Aspect','Case','Clusivity','ConjType',
                             'Definite','Degree','Deixis','DeixisRef','Echo','Evident','Foreign','Gender',
                             'Gender[dat]','Gender[erg]','Gender[obj]','Gender[psor]','Gender[subj]','Hyph',
                             'Mood','NameType','NounClass','NounType','NumForm','NumType','NumValue',
                             'Number','Number[abs]','Number[dat]','Number[erg]','Number[obj]','Number[psed]',
                             'Number[psor]','Number[subj]','PartType','Person','Person[abs]','Person[dat]',
                             'Person[erg]','Person[obj]','Person[psor]','Person[subj]','Polarity','Polite',
                             'Polite[abs]','Polite[dat]','Polite[erg]','Poss','PrepCase','PronType','PunctSide',
                             'PunctType','Reflex','Style','Subcat','Tense','Typo','VerbForm','VerbType','Voice'], 
                 vectorizer={"name":"CountVectorizer", "max_features":1000},
                 scaler={'name':'StandardScaler'}):
        
        self.config={'name':name,
                     'morph_tags':morph_tags,
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
            dictionary with feature package name as key and list of morph tags in FEATS format (e.g., {"CNG":["Case=Nom|Gender=Fem|Number=Sing", "Case=Acc|Gender=Masc|Number=Sing"]})
        """
        return {self._name: [Morphology.dict_to_feats(d) for d in [{k:v for k,v in token.morph.to_dict().items() if k in self.config['morph_tags']} for token in doc] if d]}    
         
        