import os
import pandas as pd

def get_memolon_lexicon(lang, path=''):
    """Get memolon lexicon in dictionary format
    
    Parameters
    ----------
    lang : str
        ISO-code of language to use
    path : str or dict, optional
        path to lexicon
        default loads the lexicon for a specific language (following its ISO code) from folder lang_data/emotion
        for multiple languages path is a dictionary with the languages' ISO code as keys and the respective paths as values (all languages must be covered)
        if only one language is used path can be a string containing the path
    
    Returns
    -------
    dict
        dictionary of format {word : {'valence':value, 'dominance':value,...}
    
    Raises
    ------
    FileNotFoundError
        If lexicon not found
 
    Notes
    -----
    MEmoLon from:
    Buechel, Sven; Rücker, Susanna; Hahn, Udo. 2020: Learning and Evaluating Emotion Lexicons for 91 Languages. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 1202--1217.
    Download lexica from https://zenodo.org/record/3756607/files/MTL_grouped.zip?download=1 and put it under lang_data/emotion
    """
    try:
        path = path[lang]
    except TypeError: 
        path = path or os.path.join(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3]), 'lang_data','emotion',lang+'.tsv')
    try:
        return pd.read_csv(path, sep='\t', encoding='utf-8', index_col='word', keep_default_na=False).to_dict(orient='index')
    except FileNotFoundError as e:
        raise FileNotFoundError('Make sure you have downloaded the memolon lexicon for language {}. If you saved it in a different location than lang_data/emotion make sure to configure the right path under the feature package configuration.'.format(lang)) from e
    
    
def get_iforger_lexicon(path=''):
    """Get I-ForGer in dictionary format
    
    Parameters
    ----------
    path : str, optional
        path to lexicon, defaults loads the lexicon from standard path (lang_data/formality)
    
    Returns
    -------
    pandas DataFrame
        pandas DataFrame with words as index and one column with respective scores
    
    Raises
    ------
    FileNotFoundError
        If lexicon not found
        
    Notes
    -----
    I-ForGer from:
    Eder, Elisabeth, Krieg-Holz, Ulrike, Hahn, Udo 2021: Acquiring a Formality-Informed Lexical Resource for Style Analysis. Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics, April 2021, Kyiv, Ukraine, Online Event. 2028--2041.
    """
    try:
        path = path or os.path.join(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3]), 'lang_data','formality', 'de_I-ForGer.csv')
        return pd.read_csv(path, encoding='utf-8', index_col=0)
    except FileNotFoundError as e:
        raise FileNotFoundError('Make sure you have downloaded the iforger lexicon (only available for German). If you saved it in a different location than lang_data/formality make sure to configure the right path under the feature package configuration.') from e
        