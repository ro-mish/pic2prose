# pic2prose
A package that can take in images and build a corpus and produce nlp datastructures for direct use in experimentation and model training.

Take any image with text, use p2p to generate NLP datastructures ready for use in fine-tuning LLM's, generating embeddings, sentiment classification, etc.

# Installation
```
pip install pic2prose
```

Open up your favorite editor, import, and build a robust corpus.
```
from pic2prose.structures import *

# initialize the object
# may take longer if you're not using a GPU
corpus = Corp(image_path="ex1.png")

# generate co-occurrence matrix
corpus.get_co_occurrence_matrix()

# generate tf-idf matrix
corpus.get_tfidf_matrix()

# one-hot encodings
corpus.one_hot_encode()
```

# Coming Soon
Support for building corpi from URL
