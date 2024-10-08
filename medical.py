from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import PyMuPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import ctransformers

#Extract data from PDF files
def load_pdf(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyMuPDFLoader)
    documents = loader.load()

    return documents

extracted_data = load_pdf("data/")

# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

text_chunks = text_split(extracted_data)
print("length of my chunks: ", len(text_chunks))

#download embeddings model
def download_huggingface_model():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2")
    return embeddings

embeddings = download_huggingface_model()

query_result = embeddings.embed_query("Hello, how are you?")
print("length of my query result: ", len(query_result))

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

# Initialize a client
pc = Pinecone(api_key='fa012826-08d2-41a5-a61b-9c1ee273defe')

# Define index name
index_name = "medical"

# Extract just the names of the indexes
existing_index= []
for index in pc.list_indexes():
    existing_index.append(index['name'])

# Print the list of index names for debugging
print("Existing index names:", existing_index)

# Check if the index already exists
if index_name in existing_index:
    print("Index is already created.")
else:
    # Try to create the serverless index
    try:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
            deletion_protection="disabled"
        )
        print("Index created.")
    except Exception as e:
        print(f"Error creating index: {e}")

index= pc.Index(index_name)
docsearch = index.fetch([t.page_content for t in text_chunks], embeddings, index_name=index_name)