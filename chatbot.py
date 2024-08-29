import streamlit as st
from haystack.nodes import FARMReader, DensePassageRetriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.schema import Document

# Título de la aplicación
st.title("Chatbot basado en tu documento con Haystack")

# Subida del archivo
uploaded_file = st.file_uploader("Sube un documento de texto", type="txt")

if uploaded_file is not None:
    # Leer y mostrar el contenido del archivo
    document_text = uploaded_file.read().decode("utf-8")
    st.write("Contenido del documento:")
    st.text(document_text)

    # Configuración del Document Store
    document_store = InMemoryDocumentStore()

    # Crear el documento y guardarlo en el Document Store
    docs = [Document(content=document_text, meta={"name": "uploaded_document.txt"})]
    document_store.write_documents(docs)

    # Configuración del Retriever y del Reader
    retriever = DensePassageRetriever(
        document_store=document_store, 
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=False
    )
    document_store.update_embeddings(retriever)

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

    # Crear el pipeline de preguntas y respuestas
    pipe = ExtractiveQAPipeline(reader, retriever)

    # Entrada de la pregunta del usuario
    question = st.text_input("Escribe tu pregunta:")

    if st.button("Obtener respuesta"):
        if question:
            # Obtener la respuesta
            prediction = pipe.run(query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})

            # Mostrar la respuesta
            st.write("Respuesta del Chatbot:")
            if prediction['answers']:
                st.write(prediction['answers'][0].answer)
            else:
                st.write("No se encontró una respuesta adecuada en el documento.")
        else:
            st.write("Por favor, escribe una pregunta.")
