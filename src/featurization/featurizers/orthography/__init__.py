from utils.diverse_helpers import secure_division_zero

class OrthographyFeaturizer():
    """Get orthography and mediality related features

    Parameters
    ----------
    name : str
        name of the feature package
    features : list, default=['prop_upper','prop_lower','prop_title','prop_punctuation','prop_emo']
        prop_upper: proportion of upper cased words
        prop_lower: proportion of lower cased words
        prop_title: proportion of words with title case
        prop_punctuation: proportion of punctuation
        prop_emo: proportion of emojis and emoticons
        prop_contractions: proportion of contractions (only contractions with apostrophe get counted)
    scaler : dict or bool, default={"name": "MinMaxScaler}
        scaler to scale features with
        set to 'false' if you do not want to scale the features
        choose from scalers provided by sklearn (e.g. StandardScaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) 
        or MinMaxScaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler))

    Attributes
    ----------
    config : dict
        the configurations for the feature package 
    
    Notes
    -----
    define in config file under feature_packages via:
        {
            "feature_package": "orthography",
            "features": ["prop_lower","prop_punctuation"]
        }
    """    
    def __init__(self, name, features=['prop_upper','prop_lower','prop_title','prop_punctuation','prop_emo'], scaler={'name':'MinMaxScaler'}):
        self.config = {'name':name,
                       'features':features,
                       'scaler':scaler}
        

    def featurize_doc(self, doc):
        """Get features for doc 
        
        Get proportions per token
        
        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        dict
            dictionary with feature names as keys and feature values as values
        """
        data = {}
        words = [token for token in doc if self._is_token_case(token)]
        data['prop_upper'] = secure_division_zero(sum(1 for word in words if word.is_upper),len(words))
        data['prop_title'] = secure_division_zero(sum(1 for word in words if word.is_title),len(words))
        data['prop_lower'] = secure_division_zero(sum(1 for word in words if word.is_lower),len(words))          
        data['prop_punctuation'] = secure_division_zero(doc._.n_tokens_punct,doc._.n_tokens)   
        data['prop_emo'] = sum(1 for token in doc if token._.is_emoticon or token._.is_emoji) / doc._.n_tokens
        data['prop_contractions'] = sum([1 for token in doc if "'" in token.text]) / doc._.n_tokens
        return {feature:data[feature] for feature in self.config['features']}

    
    def _is_token_case(self, token):
        """Check if token relates to capitalization
        
        Parameters
        ----------
        token : spaCy token object
        
        Returns
        -------
        bool
            True if token relates to capitalization else False 
        """
        if not any([token.is_space, token.is_punct, token.is_digit, token.is_currency, token.is_quote, token.is_bracket, 
                    token.like_url, token.like_email, token.like_num, token._.is_emoji, token._.is_emoticon]):
            return True
        else:
            return False