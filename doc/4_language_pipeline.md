# Language and Pipeline

**register** builds on [spaCy](https://spacy.io/). Thus it supports all languages supported by spaCy at least for basic feature packages (e.g. [word n-grams](2_1_2_word_lemma_ngrams.md), [character n-grams](2_1_1_character_ngrams.md) or [orthography](2_2_1_orthography.md)). Several other feature packages depend on the trained pipeline and its components provided by spaCy for a specific language (e.g. [pos n-grams](2_1_3_pos_ngrams.md) needs a tagger). For supported languages and provided pipeline components see spaCy's documentation on its [models](https://spacy.io/usage/models).
Mind, that special feature packages may be only implemented for some languages (e.g. formality for German). See the documenation on the specific [feature packages](1_basic_configurations.md/## Feature Packages) for further information.

You can compare corpora from different languages with **register**, but keep in mind that this may be not meaningful for all feature packages, i.e, comparing different language corpora using word n-grams or named entities based on different label schemes will not result in informative results. 

To specify your spaCy language model set *spacy_lm* with the language model's name. If you don't specify *spacy_lm* explicitly, **register** tries to load a standard language model based on the ISO code of the language set for a corpus in combination with "\_core\_news\_sm" and loads a blank language model otherwise.
You can also define the number of processes and the batch size for spaCy's processing pipe.

Further, you can replace parts of spaCy's processing [pipeline](https://spacy.io/usage/processing-pipelines) with [costum pipeline components](https://spacy.io/usage/processing-pipelines#custom-components) incl. [costum tokenizers](https://spacy.io/usage/linguistic-features#native-tokenizers). 

* spacy_lm : str, optional
    - spaCy language model
	- defaults to loading 'ISOLangCode\_core\_news\_sm' or a blank language model
* n_process : int, default=1 
	- number of processes for spaCy's nlp pipeline to use  
* batch_size, int, default=128
    - batch size for spaCy's nlp pipeline processing
* custom\_pipeline\_components : dict, default={}
	- dictionary with configuration for custom pipeline components to replace spaCy's components


Configuration example:
````
	"en":
				{	"spacy_lm": "en_core_web_sm",
					"n_process": 8,
					"batch_size": 1000,
                    "custom_pipeline_components": {
						                            "tagger":{
						                                        "example_rule": "example_value"
						                                     }
                                                   }
				}
````


## Costum Pipeline Components

Define custom spacy pipeline components (tokenizer, tagger, parser, ner, ...) in the [custom_pipeline_components package](src/utils/custom_pipeline_components). They must be accessible in the [__init__.py](src/utils/custom_pipeline_components/__init__.py). The tokenizer must be accessible via the function "custom\_tokenizer", which takes the spaCy pipeline as parameter. All other pipeline components must be registered via the @Language.component or @Language.factory decorator under the name "custom\_" + pipeline component name (e.g. custom\_tagger).

Examples for [__init__.py](src/utils/custom_pipeline_components/__init__.py):
````
from spacy.tokenizer import Tokenizer
from spacy.language import Language

def custom_tokenizer(nlp):   
    return MyTokenizer(nlp.vocab)
    

@Language.factory("custom_tagger", default_config={"example_rule": True})
def my_tagger(nlp, name, example_rule: bool):
    return MyTagger(example_rule=example_rule)

````


