class SyntaxFeaturizer():
    """Get syntactical features (frequency-based)

    Parameters
    ----------
    name : str
        name of the feature package
    grammar : {'constituency', 'dependency'}
        the grammar syntax trees are based on, default is 'dependency'
        'dependency': occurrences of POS of head, dependency relation and POS of child, including all combinations
        'constituency': occurrences of production rules (without lexicalizations)
    pos : {'tag', 'pos}
        whether to use Universal Dependencies v2 POS tag set (pos, default) or a finer-grained POS tag set (tag)
        for 'dependency' only
    vectorizer : dict, default={"name":"CountVectorizer", "max_features":1000} 
        vectorizer to vectorize features with
        choose from CountVectorizer(https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
        or TfidfVectorizer (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) from sklearn
    scaler : dict or bool default={"name": "StandardScaler}
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
            "feature_package": "syntax"
            "pos" = "tag"
        }   
    """

    def __init__(self, name, grammar='dependency', pos='pos',
                 vectorizer={"name":"CountVectorizer", "max_features":1000},
                 scaler={'name':'StandardScaler'}):
        self.config={'name':name,
                     'grammar':grammar,
                     'pos':pos,
                     'vectorizer':vectorizer,
                     'scaler':scaler}
        self._name = name
        if grammar == 'constituency':
            self._featurize_doc = self._featurize_doc_constituency
        else:
            self._featurize_doc = self._featurize_doc_dependency
  

    def featurize_doc(self, doc):
        """Get features for doc 
        
        Call _featurize_doc which handles different grammars

        Parameters
        ----------
        doc : spaCy Doc
        
        Returns
        -------
        dict
            dictionary with feature package name as key and list of ngrams as value, e.g., {"dependency_tuples":[()]}
        """
        return {self._name:self._featurize_doc(doc)}

    def _featurize_doc_dependency(self, doc):
        """Get dependency tuples for a document

        Parameters
        ----------
        doc : spaCy Doc

        Returns
        -------
        list
            list with dependency tuples
        """
        tups = []        
        for tup in [(getattr(token.head, f'{self.config["pos"]}_'), token.dep_, getattr(token, f'{self.config["pos"]}_')) for token in doc]:
            if tup[1] == 'ROOT':
                tups.append(f'{tup[0]} {tup[1]}')
            else:
                tups.extend([' '.join(tup),f'{tup[0]} {tup[1]}',f'{tup[2]} {tup[1]}',f'{tup[0]} {tup[2]}'])
        return tups

    def _featurize_doc_constituency(self, doc):
        """Get constituency production rules for a document

        Parameters
        ----------
        doc : spaCy Doc

        Returns
        -------
        list
            list with constituency production rules
        """
        productions = []
        for sent in doc.sents:
            for const in sent._.constituents:
                const_labels = list(const._.labels)
                if const_labels:
                    chain = [const_labels[i:i+2] for i in range(len(const_labels))] # unary chains
                    if list(const._.children):
                        chain[-1].append(' '.join([list(child._.labels)[0] if child._.labels else child._.parse_string.split(' ',1)[0][1:] for child in const._.children]))   
                    else:
                        chain[-1].append(const._.parse_string.split(' ',maxsplit=len(const_labels)+1)[len(const_labels)][1:]) 
                    productions.extend(' '.join(c) for c in chain)                        
                elif list(const._.children): # for beginning node (often empty label)
                    productions.append('ROOT '+' '.join([list(child._.labels)[0] if child._.labels else child._.parse_string.split(' ',1)[0][1:] for child in const._.children]))
        return productions
