## Infinispan vectors - Python demo

A set of demo on how to use Infinispan as a vector DB in python projects

### Requirements

- Infinispan 15+ server without authentication. There's an _infinispan-noauth.xml_ config file
available for this.
- Python 3.9 environment.
- Jupiter Lab


#### ispn_embed_demo.ipynb
It's a simple demo with a trivial embedder function for color name:
gived a color name it returns as a features vector the equivalent RGB code.
An example of similarity search with Infinispan is then provided.

#### ispn_sentences_demo.ipynb
Based on _all-MiniLM-L12-v2_ model. It maps a bunch sentences an runs
some similarity search


