import re
import urllib
import warnings
from pathlib import Path
import streamlit as st
import backoff
import pandas as pd
import PyPDF2
import ratelimit
from google.api_core import exceptions
from tqdm import tqdm
from vertexai.language_models import TextGenerationModel
warnings.filterwarnings("ignore")

@st.cache_resource


def model_with_limit_and_backoff(**kwargs):
        return generation_model.predict(**kwargs)

def reduce(initial_summary, prompt_template):
    # Concatenate the summaries from the inital step
    concat_summary = "\n".join(initial_summary)

    # Create a prompt for the model using the concatenated text and a prompt template
    prompt = prompt_template.format(text=concat_summary)

    # Generate a summary using the model and the prompt
    summary = model_with_limit_and_backoff(prompt=prompt, max_output_tokens=1024).text

    return summary

st.header("LLM API", divider="rainbow")

generation_model = TextGenerationModel.from_pretrained("text-bison@001")

st.write("Using LLM - Text and PDF model")
st.subheader("Sumarização de Arquivos")

# Story premise
arquivo = st.file_uploader("Coloque aqui o arquivo",key="arquivo")

initial_prompt_template = """
Escreva as principais informações em bullet points sobre uma vaga do texto a seguir delimitada por aspas triplas.
Escreva informações como:
Nome da Vaga,
Requisitos,
Beneficios,
Responsabilidades

```{text}```

Resumo:
    """

final_prompt_template = """
        Write a concise summary in Brazilian portuguese of the following text delimited by triple backquotes.
        Return your response in bullet points which covers the key points of the text.

        ```{text}```

        BULLET POINT SUMMARY IN Brazilian Portuguese:
    """


generate_t2t = st.button("Sumarizar Arquivo", key="generate_t2t")
if generate_t2t and arquivo:
    with st.spinner("sumarizando..."):
        reader = PyPDF2.PdfReader(arquivo)
        pages = reader.pages

        # Create an empty list to store the summaries
        initial_summary = []

        # Iterate over the pages and generate a summary for each page
        for page in tqdm(pages):
            # Extract the text from the page and remove any leading or trailing whitespace
            text = page.extract_text().strip()

            # Create a prompt for the model using the extracted text and a prompt template
            prompt = initial_prompt_template.format(text=text)

            # Generate a summary using the model and the prompt
            summary = model_with_limit_and_backoff(prompt=prompt, max_output_tokens=1024).text

            # Append the summary to the list of summaries
            initial_summary.append(summary)

        response = summary

        if response:
            st.write("arquivo Sumarizado:")
            st.write(response)

