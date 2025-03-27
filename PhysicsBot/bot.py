import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# STEP 1 - setup Mistral-7B-Instruct-v0.3
# huggingface_token = os.environ.get("huggingface_token")
huggingface_token = "*************************************"
mistral_hf_id = "mistralai/Mistral-7B-Instruct-v0.3"


def load_llm(mistral_hf_id):
    llm = HuggingFaceEndpoint(
        repo_id=mistral_hf_id,
        huggingfacehub_api_token=huggingface_token,
        temperature=0.5,
        model_kwargs={"token": huggingface_token, "max_length": "512"}
    )
    return llm


# STEP 2 - connect mistral llm to faiss db
llm_prompt = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""


def set_custom_prompt(llm_prompt):
    prompt = PromptTemplate(template=llm_prompt, input_variables=[
                            "context", "question"])
    return prompt


# STEP 3 - load faiss db
faiss_db = "vector_db/faiss_db"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(faiss_db, embedding_model,
                      allow_dangerous_deserialization=True)

# STEP 4 - setup qa chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(mistral_hf_id),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(llm_prompt)}
)

# STEP 5 - run query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
