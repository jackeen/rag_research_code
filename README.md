# RAG Ingestion Filter

## Branches
The *dev* branch shows all traces and not used modules about the experiments.

The *exp-3* branch shows all piplines for experiments and removing not associated modules, 
this branch is also standardized the log names, collection names and langchain usage.
It provides one command to run all experiments.

## Run the experiments
```shell
# this makes the module as the top-level script
python -m experiments.exp_x
```

## Run the scripts
```shell
python -m scripts.pdf2md_script
```

## Experimental agents

There are several version of agents in this project, they are named by greek letters.

- Alpha, it is not used in the current(1,2 and 3) experiments, it is for the initial exploring
- Beta, is used for 1, 2 and 3 experiments
- Gamma, for the future experiments
- Delta, for the future releasable version to support demonstration

## Packages

All packages can be installed by **requirements.text** at one time, The next list provides another way to install some of them or update their versions.

- langchain
- langgraph
- langchain-community
- langchain-ollama
- langchain-openai
- qdrant_client
- pandas
- pandas-stubs (for pandas typing)
- openpyxl (for excel file writing)
- dotenv
- openai
- sentence-transformers
- scikit-learn
- marker (experiment doc convert)
- docling (doc convert)
- fastembed (sparse vector embedding)
- spacy (NLP, python -m spacy download en_core_web_sm en_core_web_trf)
- nltk
- gliner (the model for NER task to extract entities)
- hdbscan (cluster)
- dataclasses-json
- torch
- numpy
- scipy
- mistune (markdown tree parser )
- ragas (evaluation, pip install ragas==0.3.4)
