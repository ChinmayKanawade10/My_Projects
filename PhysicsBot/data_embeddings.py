from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#STEP 1 - load the data (pdf)
DATA_PATH = "data/"

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)

print("Number of pages in the pdf: ", len(documents)) #Number of pages in the pdf: 2752

#STEP 2 - create chunks of data
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)

print("Length of Text Chunks: ", len(text_chunks))

#STEP 3 - load huggingface embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#STEP 4 - store embeddings in FAISS
faiss_db = "vector_db/faiss_db"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(faiss_db)