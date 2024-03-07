## Infinispan vectors - Python demo

A set of demo on how to use Infinispan as a vector DB in python projects

### Requirements

- Python 3.9 environment
- pip
- Jupiter Lab
- Docker (or you'll need to start Infinispan before demo)

### Setup

    pip install jupyterlab
    jupyter-lab
`jupiter-lab` will open a web gui in your browser, from there you
can run the demo notebooks

#### similarity-search-demo-1.ipynb
Based on _all-MiniLM-L12-v2_ model. Relay on a 28k+ news db that you can use
to do similarity search.

#### similarity-search-demo-2.ipynb
Based on _all-MiniLM-L12-v2_ model. Performs a similarity search on a set of
random sentences. Content is store is a separate cache and referred in the vector
cache via key id.

#### question-answer-demo.ipynb
*This demo needs an openAI api key: it must be stored in a `.env` file in the form
`OPENAI_API_KEY=<your-key-here>`*

Implement a question answer bot that can answer to question about pdf AI book
provided as an input data. For this an OpenAI api key is needed.

### Run Infinispan
By default, Infinispan is run in a docker container at the start of the demo.
If Docker is not a good option, you can
[download](https://infinispan.org/download) an Infinispan 15+ and run it
with the provided infinispan-noauth.yaml configuration. Remember to skip the docker cell
when running the demo.
