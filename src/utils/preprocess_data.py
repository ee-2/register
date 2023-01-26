import glob
import os
import pandas as pd
        
def get_texts(path_input):
    """Get text names and texts to analyze 
    
    Parameters
    ----------
    path_input : str
        path to the folder containing the files to analyze 
        or path to tsv file with one column 'id' for text id and one column 'text' for the actual text
    
    Returns
    -------
    dict
        dictionary of format {file_name : file}
        
    Raises
    ------
    FileNotFoundError
        if path is neither directory nor file
    """
    if os.path.isdir(path_input):
        return {os.path.basename(file):file for file in sorted(glob.glob(os.path.join(path_input,'**','*.txt'), recursive=True))}
    elif os.path.isfile(path_input):
        return {row['id']:row['text'] for _, row in pd.read_csv(path_input, sep='\t', encoding='utf-8').iterrows()}
    else:    
        raise FileNotFoundError(f'The path to the corpus is wrong: No such file or directory "{path_input}"')
    
def iter_texts(files):
    """Iterator for texts

    Parameters
    ----------
    files : iterator
        iterator for texts or path to texts

    Yields
    -------
    str
        text of file with newline substituted by space
    """
    if os.path.isfile(next(iter(files))):
        for file in files:
            yield ' '.join(open(file, 'r', encoding='utf-8').read().split())
    else:
        for file in files:
            yield ' '.join(file.split())
