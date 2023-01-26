from statistics import mean
from spacy.matcher import Matcher
from spacy.tokens import Token
from utils.prepare_lang_data import get_memolon_lexicon

class EmotionFeaturizer():
    """Get emotion features
    
    Get emotional scores either for each token or each lemma (if lemmatizer available) included in MEmoLon lexicon

    Parameters
    ----------
    nlp : spaCy Language object
    lang : str
        ISO-code of language to use
    name : str
        name of the feature package
    level : {'document','token','sentence'}
        level to calculate formality on, default is 'document'
        for 'document' and 'sentence' the score is determined via the average of the included token scores
    path : str or dict, optional
        path to memolon lexicon
        default loads the lexicon for a specific language (following its ISO code) from folder lang_data/emotion
        for multiple languages path is a dictionary with the languages' ISO code as keys and the respective paths as values
        if only one language is used path can be a string containing the path
    features: list, default=['valence','arousal','dominance','joy','anger','sadness','fear','disgust']
        emotion features to take into account, defaults to taking all features into account
    vectorizer : dict, default={'name':'TfidfVectorizer',"use_idf":False, "norm":"l1"} 
        vectorizer to vectorize features with
        only applies if level is 'token' or 'sentence', for level 'document' no vectorizer is taken
        choose from CountVectorizer(https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
        or TfidfVectorizer (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) from sklearn
    scaler : dict or bool or None, optional
        scaler to scale features with
        set to 'false' if you do not want to scale the features
        choose from scalers provided by sklearn (e.g. StandardScaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) 
        or MinMaxScaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler))
        defaults to None, which takes StandardScaler ({'name':'StandardScaler'}) for level 'token' or 'sentence' and for level 'document' MinMaxScaler ({'name':'MinMaxScaler'})

    Attributes
    ----------
    config : dict
        the configurations for the feature package 
    _matcher : spacy Matcher object
        matcher to match words from lexicon

    Notes
    -----
    define in config file under feature_packages via:
        {
            "feature_package": "emotion"
        }   
    
    uses lexica of:
    Buechel, Sven; Rücker, Susanna; Hahn, Udo. 2020: Learning and Evaluating Emotion Lexicons for 91 Languages. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, July 2020, Online. 1202--1217.
    Download lexica from https://zenodo.org/record/3756607/files/MTL_grouped.zip?download=1 and put it under lang_data/emotion 
    """ 
    
    def __init__(self, nlp, lang, name, path='', features=['valence','arousal','dominance','joy','anger','sadness','fear','disgust'], level='document',
                 vectorizer={'name':'TfidfVectorizer',"use_idf":False, "norm":"l1"}, scaler=None):
        self.config = {'name':name,
                       'features':features,
                       'level':level}
        if level == 'document':
            self.config['scaler'] = {'name':'MinMaxScaler'} if scaler == None else scaler
            self._get_emotion = self._get_emotion_doc
        else:
            self.config['scaler'] = {'name':'StandardScaler'} if scaler == None else scaler
            self.config['vectorizer'] = vectorizer
            if level == 'token':
                self._get_emotion = self._get_emotion_tokens
            else:
                self._get_emotion = self._get_emotion_sentences  
        self._lexicon = get_memolon_lexicon(lang, path=path)
        self._matcher = Matcher(nlp.vocab)
        if 'lemmatizer' in nlp.meta['pipeline']:        
            self._matcher.add('emotionRule', [[{"LEMMA": {"IN": list(self._lexicon)}}]], on_match=self._set_label_token_lemma) 
        else:
            self._matcher.add('emotionRule', [[{"TEXT": {"IN": list(self._lexicon)}}]], on_match=self._set_label_token_text) 
        
        for feature in features:
            if not Token.has_extension(feature):
                Token.set_extension(feature, default=None)

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
        _ = self._matcher(doc)
        data = {}
        for feature in self.config['features']:
            data[feature] = self._get_emotion(doc,feature)
        return data    

    def _get_emotion_tokens(self, doc, feature):
        """Get emotion features for tokens separately

        Parameters
        ----------
        doc : spaCy doc
        feature : str
            emotion feature                                 
        Returns
        -------
        list
            list of emotion scores for tokens for the specific feature
        """ 
        return [token._.get(feature) for token in doc if not token._.get(feature) is None]

    def _get_emotion_sentences(self, doc, feature):
        """Get emotion features for sentences separately
        
        Calculate sentence emotion feature score via the average of respective emotion scores for tokens of sentence

        Parameters
        ----------
        doc : spaCy doc
                                
        Returns
        -------
        list
            list of emotion scores for sentence for the specific feature
        """ 
        emotions = [self._get_emotion_tokens(sent, feature) for sent in doc.sents]
        return [round(mean(emotion),2) for emotion in emotions if emotion]

    def _get_emotion_doc(self, doc, feature):
        """Get emotion features for doc
        
        Calculate document emotion feature score via the average of respective emotion scores for tokens of doc
        If no word in doc is found in emotion lexicons return 0.0 (does not mean neutral, but not covered)

        Parameters
        ----------
        doc : spaCy doc
        feature : str
            emotion feature                                 
        Returns
        -------
        list
            emotion score for the doc for the chosen feature
        """   
        emotions = self._get_emotion_tokens(doc, feature)
        return round(mean(emotions),2) if emotions else 0.0
                      
    def _set_label_token_lemma(self, matcher, doc, i, matches):
        """Set score of each match found by the matcher based on lemma

        Parameters
        ----------
        matcher : spaCy matcher
        doc : spaCy doc
        i : int
            index of match in matchess
        matches : list
            list of `(match_id, start, end)` tuples, describing the matches.                                     
        """
        _, start, end = matches[i]
        for token in doc[start:end]:
            for feature in self.config['features']:
                token._.set(feature, self._lexicon[token.lemma_][feature])   
                
    def _set_label_token_text(self, matcher, doc, i, matches):
        """Set score of each match found by the matcher based on token text

        Parameters
        ----------
        matcher : spaCy matcher
        doc : spaCy doc
        i : int
            index of match in matchess
        matches : list
            list of `(match_id, start, end)` tuples, describing the matches.                                     
        """
        _, start, end = matches[i]
        for token in doc[start:end]:
            for feature in self.config['features']:
                token._.set(feature, self._lexicon[token.text][feature])  