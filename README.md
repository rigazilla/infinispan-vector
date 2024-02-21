## Infinispan vectors - Python demo

A set of demo on how to use Infinispan as a vector DB in python projects

### Requirements

- Python 3.9 environment
- pip
- Jupiter Lab
- Docker (or you'll need to start Infinispan before demo)

### Setup

- pip install jupiter-lab
- OPEN_API_KEY=<yourkey> jupyter-lab
- open the demo-book you want to run, everything should be there

### Run Infinispan
Notebooks will start Infinispan server in a docker container, so no additional setup is needed.
Otherwise, if Docker is not a good option, you can download from
[Infinispan Download](https://infinispan.org/download) an Infinispan 15+ and run it
with the provided infinispan-noauth.yaml configuration. Remember to skip the docker cell
when running the demo.

#### similarity-search-demo-1.ipynb
Based on _all-MiniLM-L12-v2_ model. Relay on a 28k+ news db that you can use
to do similarity search.

#### similarity-search-demo-2.ipynb
Based on _all-MiniLM-L12-v2_ model. Performs a similarity search on a set of
random sentences. Content is store is a separate cache and referred in the vector
cache via key id.

#### question-answer-demo.ipynb
Implement a question answer bot that can answer to question about pdf AI book
provided as an input data. For this an OpenAI api key is needed.