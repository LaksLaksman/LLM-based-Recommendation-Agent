#to load to vectorstore
#imports
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import time,os
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor

### setting embedd model
ollama_embedding = OllamaEmbedding(
    # model_name="llama3.2:latest",
    model_name="mxbai-embed-large:latest",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

#set llm
llm = Ollama(model="llama3.1:latest", request_timeout=120.0)

Settings.llm = llm
Settings.embed_model = ollama_embedding



#setting pinecone

os.environ["PINECONE_API_KEY"] = "input your api key"

api_key = os.environ["PINECONE_API_KEY"]

pc = Pinecone(api_key=api_key)
index_name = "moviedata"

# Create your index (can skip this step if your index already exists)
pc.create_index(
    index_name,
    dimension=1024,     #code":3,"message":"Vector dimension 1024 does not match the dimension of the index 1536","details":[]
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

# Initialize your index
pinecone_index = pc.Index(index_name)

# Initialize VectorStore
vector_store = PineconeVectorStore(batch_size=50 ,pinecone_index=pinecone_index)



pipeline = IngestionPipeline(
    transformations=[
        # SentenceSplitter(chunk_size=1024, chunk_overlap=200),
        # TitleExtractor(),
        ollama_embedding, 
    ],
    vector_store=vector_store,
)


# load documents
documents = SimpleDirectoryReader("your folder path").load_data()

# Ingest directly into a vector db
pipeline.run(documents=documents)