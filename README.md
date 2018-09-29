# Events: Resolving Event Coreference with Supervised Representation Learning and Clustering-Oriented Regularization

*Published in the conference STAR-SEM 2018. Link to paper is to be found [here](https://sites.google.com/view/starsem2018).*


## Startup
Run the startup script (`bash startup.sh`) to get started. It will pull and download all necessary repositories and datasets, including:

* The [coreference scorer](https://github.com/conll/reference-coreference-scorers)
* The ECB+ [dataset](www.newsreader-project.eu/results/data/the-ecb-corpus/)

There are several Python package dependencies, including [Theano](http://www.deeplearning.net/software/theano/), which is what we used for our experiments.

If you seek to implement our model, I would recommend a reimplementation in PyTorch or a more well-maintained deep learning library. 


## Relevant files
The predictions made by each model have been saved in `results/`, along with the gold standard coreference chains. After switching to the scripts directory (`cd scripts/`) you can do the following to replicate the results presented in the paper. For the within and cross-doc results:
```
bash get_scores.sh MODEL_NAME.response.conll
```

For just within-doc results:
```
bash get_scores.sh ecb_plus_events_test_mention_based_WITHINDOC_.key_conll  MODEL_NAME__within.response_conll
```

### Python code
I do not currently have the time to document the Python code, but on request I can offer assistance over email. All of the code is found in `python/`. I would recommend reimplementation of the model if you seek to develop upon CORE. If you are interested primarily in the loss function and matrix derivation of CORE, check the file `python/neural_cluster_model.py` and the definition of the loss in the `prepare_model` function. Note that several of the files and functions are deprecated and were only used for preliminary experimentation.


## Contact info
Contact [Kian Kenyon-Dean](https://kiankd.github.io/) at *kian.kenyon-dean@mail.mcgill.ca* (or, [on github](https://github.com/kiankd))  for questions about this repository.




