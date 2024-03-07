from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import pickle
import os
from dotenv import load_dotenv

load_dotenv()
# Load PDF documentation
pdf_folder_path = '/home/rigazilla/ai/data/ai-pdf'
loaders = [PyPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
documents = []
for loader in tqdm(loaders):
    try:
        documents.extend(loader.load())
    except:
        pass
with open('my_documents.pkl', 'wb') as f:
    pickle.dump(documents, f)

# Split texts in documents
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

from infinispan_vector.infinispanvs import Infinispan, InfinispanVS
ispnVS = InfinispanVS()

for text in texts:
    text.metadata.update({"text": text.page_content})

# Replacing OpenAI embeddings with HuggingFace model due to vector dimension limit
#embeddings = OpenAIEmbeddings()

# Define the path to the pre-trained model you want to use
modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

vector_store = InfinispanVS.from_documents(documents=texts, embedding=embeddings, clear_old=False)

from langchain.indexes import VectorstoreIndexCreator
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":2})
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
q = ""
while q != "bye":
    if q != "":
        result = qa({"query": q})
        print(result["result"])
    q = str(input("Question> "))
print("bye")
