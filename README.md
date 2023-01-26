# **register**

**register** is a toolkit for analyzing language use patterns which characterize **re**gisters, **g**enres and **st**yles. It provides a wide range of features and covers various languages (Note that not all feature packages are supported for all languages (see [doc](doc/1_basic_configurations.md))).


# Installation

**register** requires Python >= 3.6. 

1. Clone/download the repository
2. In the current folder run:
```
pip install -r requirements.txt
```
3. Load the [spaCy language model](https://spacy.io/usage/models) (If your language is not supported by spaCy you can still use basic feature packages (e.g. [character n-grams](doc/2_10_character_ngrams.md) or [token n-grams](doc/2_11_token_ngrams.md))), e.g.:
```
python -m spacy download de_core_news_sm
``` 

Some features need further resources:
* For features based on *constituency* parse trees load the [benepar](https://github.com/nikitakit/self-attentive-parser) model for your language:
```python
import benepar
benepar.download('benepar_de2')
```
* The feature package *emotion* needs specific language data.

    * Download the [MEmoLon](https://zenodo.org/record/3756607/files/MTL_grouped.zip?download=1) lexicon (2.4 GB) and put the file for your language (e.g., `de.tsv`) into [lang_data/emotion](./lang_data/emotion).
    * Load [textblob](https://textblob.readthedocs.io) requirements:
    ```
    python -m textblob.download_corpora
    ```

# Run **register**

Run **register** with (from the `src` directory):

```
python run_register.py path/to/your/configuration_file.json
```

If you don't specify your own JSON configuration file, the `config.json` file in the `src` directory is taken.
Edit this file to your needs or create your configuration file following the [documentation](doc/1_basic_configurations.md). 
**register** provides quite a lot configuration options, such as the choice of features you want to extract from your text or different machine learning models to use.


# A Question of Style

To reproduce the results for the feature-based models used in our paper *'A Question of Style: A Dataset for Analyzing Formality on Different Levels'* use the configuration files `config_pt16.json` and `config_c18.json` in the `src` directory. Edit the path to point to your local copy of [in_formal sentences](https://github.com/ee-2/in_formal_sentences/blob/master/in_formal_sentences.tsv).
(Attention: Constituency parsing for the PT18 model takes time. It may take a while.)


# Citation

When using **register**, please cite:

```
@inproceedings{eder-etal-2023,
    title = "A Question of Style: A Dataset for Analyzing Formality on Different Levels",
    author = "Eder, Elisabeth  and
      	      Krieg-Holz, Ulrike  and
      	      Wiegand, Michael",
    booktitle = "Findings of the Association of Computational Linguistics: EACL 2023",
    publisher = "Association for Computational Linguistics",
    note = "to appear"
}
```

**register** builds on external resources. If you use them, please cite these resources appropriately (see the [documentation](doc)).
