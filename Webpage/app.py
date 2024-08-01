from flask import Flask, render_template, request
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load the GROQ And OpenAI API KEY
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)

# Initialize LangChain Components
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Define a function to prepare vector embeddings
def vector_embedding():
    if 'vectors' not in app.config or app.config['vectors'] is None:
        print("Initializing vector store...")
        app.config['embeddings'] = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        loader = PyPDFDirectoryLoader("./data")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:20])
        vectors = FAISS.from_documents(final_documents, app.config['embeddings'])
        app.config['vectors'] = vectors
        print("Vector store initialized.")
    else:
        print("Vector store already initialized.")


@app.route('/', methods=['GET', 'POST'])
def index():
    response_text = ""
    document_chunks = []

    if request.method == 'POST':
        question = request.form.get('question', '')
        if 'embed_documents' in request.form:
            vector_embedding()
            response_text = "Vector Store DB Is Ready"
        elif question:
            if app.config['vectors'] is None:
                response_text = "Please embed documents first."
            else:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = app.config['vectors'].as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                response = retrieval_chain.invoke({'input': question})
                response_text = response['answer']
                document_chunks = response["context"]

    return render_template('index.html', response=response_text, chunks=document_chunks)


if __name__ == '__main__':
    vector_embedding()
    app.run(debug=True)
