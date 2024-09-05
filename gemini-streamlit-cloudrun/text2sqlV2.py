import os
import streamlit as st
from vertexai.preview.generative_models import (Content,
                                                GenerationConfig,
                                                GenerativeModel,
                                                GenerationResponse,
                                                Image,
                                                HarmCategory,
                                                HarmBlockThreshold,
                                                Part)
import vertexai
from google.cloud import bigquery
import numpy as np
import pandas as pd
from vertexai.language_models import TextGenerationModel
import re
from json import loads, dumps


PROJECT_ID = os.environ.get('GCP_PROJECT') #Your Google Cloud Project ID
LOCATION = os.environ.get('GCP_REGION')   #Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)
client = bigquery.Client(project=PROJECT_ID)

BQ_PROJECT_ID = "prj-p-ucbr-prod-ia-6ae3"  # @param {type:"string"}
BQ_LINKED_DATASET = "demoRAGQaRaiaDrogasil"  # @param {type:"string"}
BQ_PROCESSED_DATASET = "Teste_txt2sql"  # @param {type:"string"}
MODEL_ID = "text-bison@001" # @param {type:"string"}

BUCKET_ID = "csa-datasets-public"  # @param {type:"string"}
FILENAME = "SQL_Generator_Example_Queries.csv"  # @param {type:"string"}
client = bigquery.Client(project=BQ_PROJECT_ID)
BQ_MAX_BYTES_BILLED = pow(2, 30)  # 1GB

model = TextGenerationModel.from_pretrained(MODEL_ID)
MODEL_ID2 = "gemini-pro" # @param {type:"string"}

model2 = GenerativeModel(MODEL_ID2)
table_name = 'SalesRaiaDrogasilOBT'


QUERY = f"""\
SELECT
    '[Schema (values)]: ' || STRING_AGG(table_values, ' | ') || ';' AS tables_definition,
    '[Column names (type)]: ' || STRING_AGG(column_names_types) || ';' AS columns_definition
FROM (
    SELECT
      table_name,
      table_name || ' : ' || STRING_AGG(column_name, ' , ') as table_values,
      STRING_AGG(table_name || ' : ' || column_name || ' (' || data_type || ')', ' | ') as column_names_types
    FROM `{BQ_PROJECT_ID}.{BQ_LINKED_DATASET}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
    GROUP BY table_name
    ORDER BY table_name
)
"""

# Create query job
query_job = client.query(QUERY)
# Get first row
schema = next(query_job.result())

# Build schema definition
schema_definition = f"""\
{schema.tables_definition}

{schema.columns_definition}
"""

one_shot_template = """
Question: {question}

Answer: {query}
"""



df = pd.read_csv("gs://demo_rag_qa/0006_demoRD/0006_demoRD_teste.csv", header=0)

train_df = df.loc[df["Dataset"] == "Train", ["Question", "SQL Query"]]
eval_df = df.loc[df["Dataset"] == "Eval", ["Question", "SQL Query"]]

few_examples = ""
for index, row in train_df.iterrows():
    few_examples += one_shot_template.format(
        question=row["Question"], query=row["SQL Query"]
    )

print(f"Added {str(train_df.shape[0])} pairs as few-shot examples")


# Strip text to include only the SQL code block with
def sanitize_output(text: str) -> str:
    # Strip whitespace and any potential backticks enclosing the code block
    text = text.strip()
    regex = re.compile(r"^\s*```(\w+)?|```\s*$")
    text = regex.sub("", text).strip()

    # Find and remove any trailing quote without corresponding opening quote
    if re.search(r'^[^"]*"$', text):
        text = text[:-1]
    # Find and remove any leading quote without corresponding closing quote
    if re.search(r'^"[^"]*$', text):
        text = text[1:]

    return text

# Call model using prompt and pre-defined parameters
def generate_sql(
    model, prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
    top_k: int = 40,
    top_p: float = 0.8
) -> str:
    response = model.predict(
        prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_k=top_k,
        top_p=top_p,
    )
    text = response.text
    # Strip text to include only the SQL code block
    text = sanitize_output(text)
    print("Response stripped:")
    print(text)
    return text

def execute_sql(query: str):
    # Qualify table names with your project and dataset ID
    query = query.replace(
        table_name, f"`{BQ_PROJECT_ID}.{BQ_LINKED_DATASET}.{table_name}`"
    )
    # Validate the query by performing a dry run without incurring a charge
    job_config = bigquery.QueryJobConfig(use_query_cache=False, dry_run=True)
    try:
        response = client.query(query, job_config=job_config)
    except Exception as e:
        return e
    # Execute the query
    job_config = bigquery.QueryJobConfig(
        use_query_cache=False, maximum_bytes_billed=BQ_MAX_BYTES_BILLED
    )
    try:
        response = client.query(query)
        df = response
    except Exception as e:
        return e
    return df

def get_gemini_pro_text_response( model: GenerativeModel,
                                  contents,
                                  generation_config: GenerationConfig,
                                  stream=True):

    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }


    responses = model.generate_content(contents,
                                       generation_config = generation_config,
                                       safety_settings=safety_settings,
                                       stream=True)

    final_response = []
    for response in responses:
        try:
            # st.write(response.text)
            final_response.append(response.text)
        except IndexError:
            # st.write(response)
            final_response.append("")
            continue
    return " ".join(final_response)

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.93,
    top_k=27,
    candidate_count=1,
    max_output_tokens=2048,
    )

st.header("Vertex AI Gemini API TEXT2SQL", divider="rainbow")

question = st.text_input("Faça sua pergunta \n\n",key="question",value="Qual é a venda da regiao NORTE por vendedor?")

prompt_template = f"""
This is a task converting text into GoogleSQL statement.
We will first give you the dataset schema and then ask a question in text.
You are asked to generate SQL statement which is valid for BigQuery.
Remove any delimiters around answer such as "```"

BigQuery tables schema definition:
{schema_definition}
Here are a few shot examples:
{few_examples}
Write GoogleSQL query for following question: {question}
Answer: "Query here"
"""

generate_t2t = st.button("Me Responda", key="generate_answer")

@st.cache_data
def resultado_json(response):
    result = response.to_json(orient="split")
    return result

@st.cache_data
def resultado_df(response):
    return response


if generate_t2t and question:
    second_tab1, second_tab2 = st.tabs(["Resposta", "Prompt"])
    with st.spinner("Gerando sua resposta..."):
        with second_tab1:
            query = generate_sql(
            model,
            prompt_template.format(
                schema_definition=schema_definition,
                few_examples=few_examples,
                question=question,
            ),
            )
            response = execute_sql(query)
            step = response.to_dataframe()
            result = resultado_json(step)
            questao = """
            Com base na 'response' encontre uma relação de causa e efeito entre os resultados apontados, levando em consideração PRINCIPALMENTE O DATASET, ou seja, as informações encontradas na resposta em si. Portanto, tratando-se de uma farmácia, não vendemos sorvetes, por exemplo. Se atente nos nomes das COLUNAS Além disso:
            - Senso comum.
            - Estação do ano em que os produtos foram vendidos, se existir a informação do mês, explicando sobre sazonalidade.
            - Para que serve o produto e por quê o público o compra.
            - Tendência de crescimento ou queda das vendas se existir uma tendência clara, para encontrar correlações da venda com outro acontecimento.
            - O que é o produto.
            - Se o mês da venda era de férias, festivo, de verão, inverno ou outros. Em resumo, qualquer coisa que represente algo diferente ou especial.
            - Se existir a informação de região, informar qual é a região e suas particularidades.
            Retorne sua resposta em pontos que julgar relevantes. Me traga no máximo 5 pontos.
            PONTOS:
            """
            contents = [
                questao,
                result
                ]
            response2 = get_gemini_pro_text_response(
                                model2,
                                contents,
                                generation_config=generation_config,
                            )
            if response:
                st.write("Sua resposta:")
                st.write(step)
                st.write("Sua analise:")
                st.write(response2)
        with second_tab2:
            st.text(query)