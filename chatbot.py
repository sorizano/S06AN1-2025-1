import streamlit as st
from haystack.nodes import FARMReader, TransformersReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.schema import Document  # Asegúrate de importar la clase Document

# Configuración de Streamlit
st.title("Chatbot con Haystack basado en tu documento")

# Subida del archivo
uploaded_file = st.file_uploader("Sube un documento de texto", type="txt")

if uploaded_file is not None:
    # Leer el contenido del archivo subido
    document_text = uploaded_file.read().decode("utf-8")

    # Mostrar el contenido del documento
    st.write("Contenido del documento:")
    st.text(document_text)

    # Configurar Haystack
    document_store = InMemoryDocumentStore()

    # Convertir el texto en un formato compatible para Haystack usando la clase Document
    docs = [Document(content=document_text, meta={"name": "uploaded_document.txt"})]
    document_store.write_documents(docs)

    # Inicializar el modelo retriever y reader
    retriever = DensePassageRetriever(document_store=document_store, embedding_model="facebook/dpr-ctx_encoder-single-nq-base")
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

    # Pipeline de QA
    pipe = ExtractiveQAPipeline(reader, retriever)

    # Entrada de la pregunta del usuario
    question = st.text_input("Escribe tu pregunta:")

    if st.button("Obtener respuesta"):
        if question:
            # Obtener la respuesta
            prediction = pipe.run(query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
            
            # Mostrar la respuesta
            st.write("Respuesta del Chatbot:")
            if len(prediction['answers']) > 0:
                st.write(prediction['answers'][0].answer)
            else:
                st.write("No se encontró una respuesta adecuada en el documento.")
        else:
            st.write("Por favor, escribe una pregunta.")
