from textblob_de import TextBlobDE as TextBlob

class SentimentFeaturizer():
    """Get sentiment for German
    
    Used in combination with emotion featurizer, not as standalone feature package (no own config)

    Parameters
    ----------
    features : list, default=['polarity']
        polarity: polarity score
    level : {'token','sentence','document'}
        level to calculate formality on, default is 'token'

    Notes
    -----
    Uses polarity score from TextBlob-de (subjectivity not implemented yet) (https://textblob-de.readthedocs.io/en/latest/)
    """
    def __init__(self, features=['polarity'], level='document'):
        self._features = features
        if level == 'token':
            self._get_sentiment = self._get_sentiment_tokens
        elif level == 'sentence':
            self._get_sentiment = self._get_sentiment_sentences
        else:
            self._get_sentiment = self._get_sentiment_doc

    def featurize_doc(self, doc):
        """Get features for doc 

        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        dict
            dictionary with feature names as keys, 
            values are either list of sentiment scores (for level 'token' or 'sentence') or sentiment score itself (for level 'document') for the specific feature
        """ 
        return self._get_sentiment(doc)
    
    def _get_sentiment_tokens(self, doc):
        """Get sentiment for tokens separately

        Parameters
        ----------
        doc : spaCy doc
                                
        Returns
        -------
        dict
            dictionary with the name of the features as keys with lists of sentiment scores for tokens for the specific feature as values
        """  
        data = {}
        sentiments = [TextBlob(token.text) for token in doc]
        for feature in self._features:
            data[feature] = [getattr(token,feature) for token in sentiments]  
        return data

    def _get_sentiment_sentences(self, doc):
        """Get sentiment for sentences separately

        Parameters
        ----------
        doc : spaCy doc
                                
        Returns
        -------
        dict
            dictionary with the name of the features as keys with lists of sentiment scores for sentences for the specific feature as values
        """ 
        data = {}
        sentiments = [TextBlob(sent.text) for sent in doc.sents]
        for feature in self._features:
            data[feature] = [getattr(sent, feature) for sent in sentiments]      
        return data

    def _get_sentiment_doc(self, doc):
        """Get sentiment for doc

        Parameters
        ----------
        doc : spaCy doc
                                
        Returns
        -------
        dict
            dictionary with the name of the features as keys with the respective sentiment score per doc as values
        """ 
        data = {}
        sentiment = TextBlob(doc.text).sentiment
        for feature in self._features:
            data[feature] = getattr(sentiment, feature) 
        return data
    