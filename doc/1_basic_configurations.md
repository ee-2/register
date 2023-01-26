# Basic Configurations

## Mode
First, choose what you want to do with your texts.

* *analyze* or empty: If you only want to analyze your texts, choose *analyze* or leave the mode empty. Your texts' features are dumped in a file in tsv format.
* *cluster*: If you want to find clusters of similar texts in your corpus, choose *cluster*.
* *train\_classifier*: If you have texts associated with to two or more classes and want to train or evaluate a classifier on them, choose *train\_classifier*. For further instructions on how to train a  model, see [models](3_models.md).
* *classify*: If you have already trained a classifier, you can label a text with unknown class by choosing *classify*. For instructions on how to load a pretrained model, see [models](3_models.md).
* *train\_linear\_regressor*: If you have texts associated with continuous scores and want to train or evaluate a linear regressor on them, choose *train\_linear\_regressor*. For further instructions on how to train a  model, see [models](3_models.md).
* *score*: If you have already trained a linear regression model, you can score a text with unknown rating by choosing *score*. For instructions on how to load a pretrained model, see [models](3_models.md). 


## Corpora
Next, specify the texts or corpora you want to process with **register**. 
You can use multiple corpora. For each one you have to define the path (*path*) to the respective folder containing txt files or to a tsv file which contains one column 'id' for the text ids and one column 'text' for the actual texts. Further define the corpus language (*lang*). **register** lets you compare corpora from different languages, but keep in mind that this may be not meaningful for all feature packages (e.g. for word n-grams). If you want to name your corpus different than it's folder name, do it via *name* (pay attention to not having multiple corpora with the same name).

* *path*: path to corpus, folder with text files or tsv file (obligatory!)
* *lang*: ISO language code of the corpus (obligatory!)
* *name*: corpus name (optional)

Further, you can specify a class associated with the corpus (*class*) or point to a tsv file containing the text id and class (or score for linear regression) via *path\_targets*.
You can define train/test splits either via setting *set* to train, test or apply or point to a tsv file containing the set for each file. For further instructions see [models](3_models.md).

* *class*: class/label associated with a corpus
* *path\_targets*: path to a tsv file containing the text id (e.g., document name) and the associated classes or scores in class/score
* *set*: train, test or apply sets for training a model
* *path\_train\_test\_splits*: path to a tsv file containing the text id and the associated sets (train or test)


## Feature Packages
There are different feature packages you can choose from:

* [character n-grams](2_10_character_ngrams.md)
* [token n-grams](2_11_token_ngrams.md) 
* [span n-grams](2_12_span_ngrams.md)
* [orthography](2_13_orthography.md)
* [metrics](2_14_metrics.md)
* [morphology](2_15_morphology.md)
* [syntax](2_16_syntax.md)
* [named entities](2_17_named_entities.md)
* [embeddings](2_18_embeddings.md)
* [emotion](2_19_emotion.md)
* [formality](2_20_formality.md) (for German)


# Further Configurations

## Output
You can specify the path to folder for the output of **register** in *base_dir* (default is the folder you are running the program from).
If you train a model you get explanations for the model's output using [SHAP](https://github.com/slundberg/shap) by setting *explanation_shap* to *true*.

## Models
For chooseable machine-learning models see [models](3_models.md).

## Languages
For specific configurations regarding the chosen languages see [language](4_language_pipeline.md).

## Vectorizing and Scaling
For chooseable vectorizers and scaling options see [vectorizers](5_vectorizers.md) and [scaling](6_scaling.md).

# Examples
Configure **register** via the `config.json` file in the `src` directory or via another json file, which you have to pass as argument to register. See [simple_config.json](examples/simple_config.json) or [advanced_config.json](examples/advanced_config.json) for a simple or advanced configuration example. Other examples are [config_pt16.json](../src/config_pt16.json) or [config_c18.json](../src/config_c18.json).
