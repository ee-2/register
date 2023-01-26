from statistics import mean
from collections import Counter

class MetricsFeaturizer():
    """Get various metrics (readability, formality, syntax)

    Parameters
    ----------
    nlp : spaCy Language object
    name : str
        name of the feature package
    features : list, default=['avg_token_length', 'avg_sentence_length_tokens', 'avg_sentence_length_characters','avg_dependency_height', 'flesch_reading_ease', 'heylighen_f']
        avg_token_length: average length of tokens
        avg_sentence_length_tokens: average length of sentences in tokens
        avg_sentence_length_characters: average length of sentences in characters
        avg_constituency_height: average height of constituency trees of sentences, normalized by sentence length
        avg_dependency_height: average height of dependency trees of sentences, normalized by sentence length
        flesch_kincaid_grade_level: Flesch-Kincaid grade level (J. Peter Kincaid, Robert P. Fishburne Jr., Richard L. Rogers, and Brad S. Chissom. 1975: Derivation of new readability formulas (automated readability index, fog count and Flesch reading ease formula) for navy enlisted personnel. Technical report, DTIC Document.)
        flesch_reading_ease: Flesch reading ease (Flesch, Rudolf. 1948: A new readability yardstick. Journal of Applied Psychology 32:221-33.)
        heylighen_f: formality score as defined by Heylighen, Francis and Dewaele Jean-Marc. 1999: Formality of language: definition, measurement and behavioral determinants. Technical report, Center ”Leo Apostel”, Free University of Brussels.    
    scaler : dict or bool, default={"name": "MinMaxScaler}
        scaler to scale features with
        set to 'false' if you do not want to scale the features
        choose from scalers provided by sklearn (e.g. StandardScaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) 
        or MinMaxScaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler))
    
    Notes
    -----
    define in config file under feature_packages via:
        {
            "feature_package": "metrics",
            "features": ["avg_dependency_height",
                        "heylighen_f"]
        }
    """ 

    def __init__(self, nlp, name, features=['avg_token_length', 'avg_sentence_length_tokens', 'avg_sentence_length_characters', 
                                            'avg_dependency_height', 'flesch_reading_ease','heylighen_f'], scaler={'name':'MinMaxScaler'}):
        self.config = {'name':name,
                       'features':features,
                       'scaler':scaler}
            
        self._non_deictic = set(nlp.vocab.strings.add(label) for label in ['NOUN', 'PROPN', 'ADJ', 'DET', 'ADP'])
        self._deictic = set(nlp.vocab.strings.add(label) for label in ['VERB', 'AUX', 'ADV', 'INTJ', 'PRON'])
        self._tags = self._non_deictic.union(self._deictic)

        self._featurize_doc = self._featurize_doc_constituency if 'avg_constituency_height' in features else self._featurize_doc_default
            
    def featurize_doc(self, doc):
        """Get features for doc 
        
        Call _featurize_doc which handles different grammars
        
        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        dict
            dictionary with feature names as keys and feature values as values
        """
        data = self._featurize_doc(doc)
        return {feature:data[feature] for feature in self.config['features']}

    def _featurize_doc_default(self, doc):
        """Get features for doc (default)
        
        Does not calculate constituency based metrics
        
        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        dict
            dictionary with feature names as keys and feature values as values
        """
        data = {}
        data['avg_token_length'] = doc._.avg_token_len_chars
        data['avg_sentence_length_tokens'] = doc._.avg_sent_len_tokens
        data['avg_sentence_length_characters'] = doc._.avg_sent_len_chars
        data['avg_dependency_height'] = mean([self._get_dependency_tree_height(sent.root)/len(sent) for sent in doc.sents])
        data['flesch_kincaid_grade'] = self._get_flesch_kincaid_grade(doc)
        data['flesch_reading_ease'] = self._get_flesch_ease(doc)
        data['heylighen_f'] = self._get_heylighen_formality_score(doc) 
        return data

    def _featurize_doc_constituency(self, doc):
        """Get features for doc with constituency based metrics
        
        Separately for efficiency reasons
        
        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        dict
            dictionary with feature names as keys and feature values as values
        """
        data = self._featurize_doc_default(doc)
        data['avg_constituency_height'] = mean([self._get_constituency_tree_height(sent)/len(sent) for sent in doc.sents])
        return data
    

    def _get_dependency_tree_height(self, node):
        """Get height of dependency tree
        
        Parameters
        ----------
        node : spaCy Token
        
        Returns
        -------
        int
            height of dependency tree

        Notes
        -----
            code adapted from https://stackoverflow.com/questions/35920826/how-to-find-height-for-non-binary-tree
        """
        if not list(node.children):
            return 0
        else:
            return 1 + max(self._get_dependency_tree_height(x) for x in node.children)

    def _get_constituency_tree_height(self, span):
        """Get height of constituency tree (with lexicalizations)
        
        Examples (incl. unary chain):
        
        (NP (PP (PP (NN Haus)))) -> return 4   
        (NP (PP (PP (Det Das) (NN Haus)))) -> return 4 (first count labels (3), then terminal production (1))
        
        Parameters
        ----------
        span : spaCy Span
        
        Returns
        -------
        int
            height of constituency tree
        """
        if not list(span._.children):  
            # terminal productions: count labels to account for unary chains (one constituent)     
            return len(list(span._.labels))+1 if list(span._.labels) else 1 
        else:
            # 1 if span has no labels (at the beginning) 
            return (len(list(span._.labels)) if list(span._.labels) else 1) + max(self._get_constituency_tree_height(x) for x in span._.children)

    def _get_heylighen_formality_score(self, doc):
        """Roughly calculate F-score (formality score) 
        
        POS tags based on Universal Dependencies v2 POS tag set for default
        
        F-Score defined by:
        Heylighen, Francis; Dewaele, Jean-Marc. 1999. Formality of Language: definition, measurement and behavioral determinants. 
        Technical Report Internal Report, Center ”Leo Apostel”, Free University of Brussels, Brussels. 

        Parameters
        ----------
        doc : spaCy Doc or spaCy Span object for a single sentence

        Returns
        -------
        float 
            score between 0 and 100
            0 if no F-Score could be calculated (F-Score never reaches this limit)       
        """
        pos_counts = Counter([token.pos for token in doc if token.pos in self._tags])
        total_words = sum(v for v in pos_counts.values())
        return (((sum(v for k, v in pos_counts.items() if k in self._non_deictic) - sum(v for k,v in pos_counts.items() if k in self._deictic)
                  )/total_words*100) + 100) / 2 if total_words else 0
                  
    def _get_flesch_kincaid_grade(self, doc):
        """Calculate Flesch-Kincaid grade

        Parameters
        ----------
        doc : spaCy Doc or spaCy Span object for a single sentence

        Returns
        -------
        float 
            score > -3.4
        """
        words = [token for token in doc if token._.syllables_count]
        return (0.39 * (len(words) / doc._.n_sents)) + (11.8 * (sum(token._.syllables_count for token in words) / len(words))) - 15.59 if words else 0


    def _get_flesch_ease(self, doc):
        """Calculate Flesch reading ease

        Parameters
        ----------
        doc : spaCy Doc or spaCy Span object for a single sentence

        Returns
        -------
        float 
            score between 0 and 100
        """
        words = [token for token in doc if token._.syllables_count]
        return 206.835 - (1.015 * (len(words) / doc._.n_sents)) - (84.6 * (sum(token._.syllables_count for token in words) / len(words))) if words else 0