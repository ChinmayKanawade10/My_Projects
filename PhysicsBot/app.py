import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# STEP 1 - load faiss db and mistral llm

# huggingface_token = os.environ.get("huggingface_token")
huggingface_token = "*************************************"
mistral_hf_id = "mistralai/Mistral-7B-Instruct-v0.3"

faiss_db = "vector_db/faiss_db"


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(faiss_db, embedding_model,
                          allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=[
                            "context", "question"])
    return prompt


def load_llm(mistral_hf_id, huggingface_token):
    llm = HuggingFaceEndpoint(
        repo_id=mistral_hf_id,
        huggingfacehub_api_token=huggingface_token,
        temperature=0.5,
        model_kwargs={"token": huggingface_token, "max_length": "512"}
    )
    return llm

# STEP 2 - streamlit user interface


def main():
    st.title("🤖 PHYSICS BOT")
    st.subheader("Ask any physics-related queries !")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask your query !")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        llm_prompt = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(mistral_hf_id, huggingface_token),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(llm_prompt)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append(
                {'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
