from collections import Counter  
from utils.diverse_helpers import secure_division_zero      
from models.linearregression.linear_regression_scorer import LinearRegression_scorer
import os
from utils.prepare_lang_data import get_iforger_lexicon
from statistics import mean
from spacy.matcher import Matcher
from spacy.tokens import Token
import numpy as np
from flair.data import Sentence
from flair.embeddings import BytePairEmbeddings, WordEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings

class FormalityFeaturizer():
    """Get formality scores for German

    Calculate formality scores for tokens, sentences and documents 

    Parameters
    ----------
    nlp : spaCy Language object
    name : string
        name of the feature package
    level : {'token','sentence','document'}
        level to calculate formality on, default is 'document'
        for 'document' and 'sentence' the score is determined via the average of the included token scores
    path : string, optional
        path to iforger lexicon (or similar lexicon)
        default loads the lexicon from folder lang_data/formality
    vectorizer : dict, default={'name':'TfidfVectorizer',"use_idf":False, "norm":"l1"} 
        vectorizer to vectorize features with,
        for level 'document' no vectorizer is taken
        choose from CountVectorizer(https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
        or TfidfVectorizer (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) from sklearn
    scaler : dict or bool or None, optional
        scaler to scale features with
        set to 'false' if you do not want to scale the features,
        choose from scalers provided by sklearn (e.g. StandardScaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) 
        or MinMaxScaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler))
        defaults to None, which takes StandardScaler ({'name':'StandardScaler'}) for level 'token' or 'sentence' and for level 'document' MinMaxScaler ({'name':'MinMaxScaler'})

    Attributes
    ----------
    config : dict
        the configurations for the feature package 
    _matcher : spacy Matcher object
        matcher to match autosemantica
    _iforger_model : LinearRegression_scorer
        linear regression model to score words for formality
        
    _embeddings : DocumentPoolEmbeddings
        flair embeddings; we stack fastText embeddings with BPEmb for OOV functionality
            
    Notes
    -----
    define in config file under feature_packages via:
        {
            "feature_package": "formality",
            "name": "iforger"
        }    
    
    uses I-ForGer:
    Eder, Elisabeth, Krieg-Holz, Ulrike, Hahn, Udo 2021: Acquiring a Formality-Informed Lexical Resource for Style Analysis. Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics, April 2021, Kyiv, Ukraine, Online. 2028--2041.
    """   
    def __init__(self, nlp, name, path='', level='document', 
                 vectorizer={'name':'TfidfVectorizer',"use_idf":False, "norm":"l1"}, scaler=None):
        self.config = {'name':name,
                       'level':level}
        
        if level == 'document':
            self.config['scaler'] = {'name':'MinMaxScaler'} if scaler == None else scaler
            self._get_formality = self._get_formality_doc
        else:
            self.config['scaler'] = {'name':'StandardScaler'} if scaler == None else scaler
            self.config['vectorizer'] = vectorizer
            if level == 'sentence':
                self._get_formality = self._get_formality_sentences
            else:
                self._get_formality = self._get_formality_tokens
                            
        # just score tokens matched by matcher
        self._matcher = Matcher(nlp.vocab)
        self._matcher.add('Word2ScoreRule', 
                        [[{'IS_SPACE':False, 'IS_PUNCT':False, 'IS_DIGIT':False, 'IS_CURRENCY':False, 'LIKE_URL':False, 'LIKE_EMAIL':False, 'LIKE_NUM':False, 'IS_QUOTE':False, 'IS_BRACKET':False,
                           'TAG': {'IN': ['ADJA','ADJD','NN','NE','NNE','VVFIN','VVINF','VVIZU','VVIMP','VVPP','ADV','PROAV','PWAV']},
                           '_':{'is_emoji':False, 'is_emoticon':False}}]])  
        if not Token.has_extension('iforger_score'):
            Token.set_extension('iforger_score', default=None)
        self._embeddings = DocumentPoolEmbeddings(StackedEmbeddings(embeddings=[WordEmbeddings('de-crawl'), BytePairEmbeddings('de',dim=100,syllables=100000)]))
        self._iforger_model = self._load_scorer(nlp, path=path)

    def featurize_doc(self, doc):
        """Get features for doc 

        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        dict
            dictionary with feature package name as keys, 
            value is either list of scores (for level 'token' or 'sentence') or score itself (for level 'document')
        """ 
        self._set_scores_tokens(doc, self._matcher(doc), 'iforger_score') 
        return {self.config['name'] : self._get_formality(doc, 'iforger_score')} 
    
    def _load_scorer(self, nlp, path):
        """Set scorer model 
        Loads it if already trained, else trains model and saves it
        
        Parameters
        ----------
        nlp : spaCy Language object
        path : string
            path to I-ForGer lexicon
                                
        Returns
        -------
        LinearRegression_scorer
            formality scoring model
        """
        print('Loading/Training Formality Word Scorer...')
        dirPath = os.path.join(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-7]), 'lang_data','formality')
        if os.path.exists(os.path.join(dirPath, 'de_{}_I-ForGer_model.pickle'.format(nlp.meta['name']))):
            return LinearRegression_scorer(path=os.path.join(dirPath, 'de_{}_I-ForGer'.format(nlp.meta['name'])))
        else:
            seeds = get_iforger_lexicon(path=path)
            iforger_model = LinearRegression_scorer()
            token_embs = [Sentence(s) for s in seeds.index]
            self._embeddings.embed(token_embs)
            iforger_model.fit(np.array([token_emb.embedding.tolist() for token_emb in token_embs]), seeds['Score'])
            iforger_model.dump('de_{}_I-ForGer'.format(nlp.meta['name']),dirPath)
            return iforger_model
    
    def _set_scores_tokens(self, doc, matches, attr):
        """Set score of each match found by the matcher

        Predict score for each token, embeddings can be context dependent
         
        Parameters
        ----------
        doc : spaCy doc
        matches : list
            list of `(match_id, start, end)` tuples, describing the matches.
        attr : string
            attribute to set
                                
        Returns
        -------
        None       
        """
        token_embs = [Sentence(token.text) for _, start, end in matches for token in doc[start:end]]
        if token_embs:
            self._embeddings.embed(token_embs)
            for token, score in zip([token for _, start, end in matches for token in doc[start:end]], self._iforger_model.predict(np.array([token_emb.embedding.tolist() for token_emb in token_embs]))):
                token._.set(attr, round(score,2))
    
    def _get_formality_tokens(self, doc, attr):
        """Get formality for tokens separately

        Parameters
        ----------
        doc : spaCy doc
        attr : string
            attribute to set
                                
        Returns
        -------
        list
            list of formality scores for scored tokens
        """        
        return [token._.get(attr) for token in doc if not token._.get(attr) is None]

    def _get_formality_sentences(self, doc, attr):
        """Get formality for sentences separately 
        
        Calculate sentence formality score via the average of formality scores for tokens of sentence
        
        Parameters
        ----------
        doc : spaCy doc
        attr : string
            attribute to set
                                
        Returns
        -------
        list
            list of formality scores for sentence
        """
        formalities = [self._get_formality_tokens(sent, attr) for sent in doc.sents]
        return [round(mean(formality),2) for formality in formalities if formality]

    def _get_formality_doc(self, doc, attr):
        """Get formality for doc 
        
        Calculate document formality score via the average of formality scores for tokens of doc
        If no word formality scores computed for doc return 0.0 (neutral)
        
        Parameters
        ----------
        doc : spaCy doc
        attr : string
            attribute to set
                                
        Returns
        -------
        float
            formality score of document
        """
        formalities = self._get_formality_tokens(doc, attr)
        return round(mean(formalities),2) if formalities else 0.0
