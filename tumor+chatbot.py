import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers

# Initialize Pinecone
PINECONE_API_KEY = "2bd5fd9f-2c56-42f9-aa1b-037960262fff"
PINECONE_API_ENV = "gcp-starter"
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Load the trained brain tumor detection model
tumor_model = load_model('Brain Tumor detection.g9')

# Function to make tumor predictions
def predict(image):
    img = Image.open(image)
    img = img.resize((128, 128))  # Resize image to match model input size
    img = np.array(img) / 255.0  # Normalize the image
    input_img = np.expand_dims(img, axis=0)  # Expand dimensions for model input
    res = tumor_model.predict(input_img)
    return res

# Load PDF and create embeddings for the chatbot
@st.cache_resource
def load_and_process_data():
    # Extract data from the PDF
    loader = DirectoryLoader("data/", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Create text chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)

    # Download embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create embeddings and store in Pinecone
    index_name = "medical-chatbot"
    docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

    return docsearch

# Load data for the chatbot
docsearch = load_and_process_data()

# Set up the LLM and prompt template for the chatbot
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama", config={'max_new_tokens': 512, 'temperature': 0.8})

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs={"prompt": PROMPT}
)

# Streamlit App Layout
def main():
    st.title("🩺 Medical Applications")
    
    # Sidebar for selection
    app_mode = st.sidebar.selectbox("Choose Application", ["Chatbot", "Brain Tumor Detection"])

    if app_mode == "Chatbot":
        st.subheader("Medical Chatbot")
        st.write("Ask me anything about the medical topics in the uploaded PDFs.")

        user_input = st.text_input("Input Prompt:")
        if st.button("Submit"):
            if user_input:
                result = qa({"query": user_input})
                st.write("Response: ", result["result"])
            else:
                st.write("Please enter a question.")

    elif app_mode == "Brain Tumor Detection":
        st.subheader("Brain Tumor Detection")
        st.write("Upload an MRI image for brain tumor prediction.")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Make prediction
            prediction = predict(uploaded_file)
            if prediction[0][0] < 0.5:  # Adjust threshold based on your model's output
                st.markdown("<h2 style='color: green;'>No tumor detected.</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='color: red;'>Tumor detected!</h2>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
