import streamlit as st
from haystack.nodes import FARMReader, TransformersReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.utils import convert_files_to_docs

#Configuración de srteamlit
st.title("Chatbot con Haystack basado en un Documento")

#Subida del archivo
uploaded_file = st.file_uploader("Sube un documento de texto", type="txt")