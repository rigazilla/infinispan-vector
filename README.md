## Infinispan vectors - Python demo

A set of demo on how to use Infinispan as a vector DB in python projects

### Requirements

- Infinispan 15+ server without authentication. There's an _infinispan-noauth.xml_ config file
available for this.
- Python 3.9 environment.
- Jupiter Lab

### Setup

- python3.9 -m venv demovenv   # Create a Python 3.9 environment
- source demovenv/bin/activate # activate it
- pip install jupiter-lab
- python -m pip install .
- jupyter-lab
- open the demo-book you want to run

#### ispn_sentences_demo.ipynb
Based on _all-MiniLM-L12-v2_ model. Relay on a 40k sentences db that you can use
to do similarity search

#### ispn_news_demo.ipynb
Based on _all-MiniLM-L12-v2_ model. Relay on a 28k+ BBC news db that you can use
to do similarity search

#### ispn_embed_demo.ipynb
Old demo maybe broken
It's a simple demo with a trivial embedder function for color name:
gived a color name it returns as a features vector the equivalent RGB code.
An example of similarity search with Infinispan is then provided.


