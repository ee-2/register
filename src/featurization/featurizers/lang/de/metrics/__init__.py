from statistics import mean
from collections import Counter
from ....metrics import MetricsFeaturizer
from spacy.matcher import PhraseMatcher
from .hedges import hedge_words

class MetricsFeaturizer(MetricsFeaturizer):
    """Get various metrics (readability, formality, syntax) for German

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
        flesch_reading_ease: Flesch reading ease for German (Toni Amstad: Wie verständlich sind unsere Zeitungen? Universität Zürich: Dissertation 1978.)
        heylighen_f: formality score as defined by Heylighen, Francis and Dewaele Jean-Marc. 1999: Formality of language: definition, measurement and behavioral determinants. Technical report, Center ”Leo Apostel”, Free University of Brussels.
        prop_hedge_phrases: proportion of hedge words and phrases
        prop_first_prs_pron: proportion of first person pronouns
        prop_third_prs_pron: proportion of third person pronouns
    pos : {'tag', 'pos'}
        whether to use Universal Dependencies v2 POS tag set (pos) or the Tiger annotation scheme based on STTS (tag)
        applies only for heylighen_f
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
            "features": ["avg_dependency_height","heylighen_f"],
            "pos" : "tag"
        }
    """    

    def __init__(self, nlp, name, features=['avg_token_length', 'avg_sentence_length_tokens', 'avg_sentence_length_characters', 
                                            'avg_dependency_height', 'flesch_reading_ease','heylighen_f'], pos='pos', scaler={'name':'MinMaxScaler'}):
        self.config = {'name':name,
                       'features':features,
                       'pos':pos,
                       'scaler':scaler}

        if pos == 'tag':
            self._non_deictic = set(nlp.vocab.strings.add(label) for label in ['NN','NE',
                                                                              'ADJA','ADJD',
                                                                              'ART',
                                                                              'APPR','APPRART','APPO','APZR'])
            self._deictic = set(nlp.vocab.strings.add(label) for label in ['VVFIN','VVIMP','VVINF','VVIZU','VVPP','VAFIN','VAIMP','VAINF','VAPP','VMFIN','VMINF','VMPP',
                                                                           'ADV','PROAV',
                                                                           'PDS','PDAT','PIS','PIAT','PPER','PPOSS','PPOSAT','PRELS','PRELAT','PRF','PWS','PWAT','PWAV',
                                                                           'ITJ'])
        else:
            self._non_deictic = set(nlp.vocab.strings.add(label) for label in ['NOUN', 'PROPN', 'ADJ', 'DET', 'ADP'])
            self._deictic = set(nlp.vocab.strings.add(label) for label in ['VERB', 'AUX', 'ADV', 'INTJ', 'PRON'])
        
        self._tags = self._non_deictic.union(self._deictic)

        # hedge words    
        self._hedge_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")    
        self._hedge_matcher.add('hedgesRule', list(nlp.tokenizer.pipe(hedge_words)))
        
        self._featurize_doc = self._featurize_doc_constituency if 'avg_constituency_height' in features else self._featurize_doc_default


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
        data['prop_hedge_phrases'] = self._get_nr_hedge_phrases(doc) / doc._.n_tokens
        data['prop_first_prs_pron'] = self._get_nr_first_person_pronouns(doc) / doc._.n_tokens
        data['prop_third_prs_pron'] = self._get_nr_third_person_pronouns(doc) / doc._.n_tokens
        return data

    def _get_nr_hedge_phrases(self, doc):
        """Get number of hedge words and phrases

        Parameters
        ----------
        doc : spaCy Doc or spaCy Span object for a single sentence

        Returns
        -------
        int
        """
        return sum([1 for _ in self._hedge_matcher(doc)])

        
    def _get_heylighen_formality_score(self, doc):
        """Roughly calculate F-score (formality score)
        
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
        pos_counts = Counter([getattr(token, self.config['pos']) for token in doc if getattr(token, self.config['pos']) in self._tags])
        total_words = sum(v for v in pos_counts.values())
        return (((sum(v for k, v in pos_counts.items() if k in self._non_deictic) - sum(v for k,v in pos_counts.items() if k in self._deictic)
                  )/total_words*100) + 100) / 2 if total_words else 0   
                  

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
        return 180 - (len(words) / doc._.n_sents) - (58.5 * (sum(token._.syllables_count for token in words) / len(words))) if words else 0  
    

    def _get_nr_first_person_pronouns(self, doc):
        """Get number of first person pronouns

        Parameters
        ----------
        doc : spaCy Doc or spaCy Span object for a single sentence

        Returns
        -------
        int
        """
        return sum(1 for token in doc if token.morph.get('PronType') == ['Prs'] and token.morph.get('Person') == ['1'])


    def _get_nr_third_person_pronouns(self, doc):
        """Get number of third person pronouns

        Parameters
        ----------
        doc : spaCy Doc or spaCy Span object for a single sentence

        Returns
        -------
        int
        """
        return sum(1 for token in doc if token.morph.get('PronType') == ['Prs'] and token.morph.get('Person') == ['3'])