"""
in this package you can define custom spacy pipeline components (tokenizer, tagger, parser, ner, etc.)

see: https://spacy.io/usage/processing-pipelines#custom-components 
and for custom tokenizers: https://spacy.io/usage/linguistic-features#native-tokenizers

classes/functions must be accessible in this __init__.py:
- the tokenizer must be accessible via the function custom_tokenizer, which takes the pipeline as parameter
- all other pipeline components must be registered via the @Language.component or @Language.factory decorator under the name "custom_**component_name**" (e.g. custom_tagger)

in config custom components to replace spacy components are defined in the language configuration (i.e. "de")
for an example replacing the tagger with a custom tagger see (call can be a function or a class as explained in the spacy documentation)

"de": {
        "custom_pipeline_components": {
                                        "tagger":{
                                                    "example_rule": "example_value"
                                                }
                                      }
      }

EXAMPLES (based on the spacy documentation):

from spacy.tokenizer import Tokenizer
from spacy.language import Language

def custom_tokenizer(nlp):   
    return MyTokenizer(nlp.vocab)
    

@Language.factory("custom_tagger", default_config={"example_rule": True})
def my_tagger(nlp, name, example_rule: bool):
    return MyTagger(example_rule=example_rule)

"""