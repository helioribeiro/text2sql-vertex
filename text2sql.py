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



PROJECT_ID = os.environ.get('GCP_PROJECT') #Your Google Cloud Project ID
LOCATION = os.environ.get('GCP_REGION')   #Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)
client = bigquery.Client(project=PROJECT_ID)

BQ_PROJECT_ID = "devhelio"  # @param {type:"string"}
BQ_LINKED_DATASET = "pharma"  # @param {type:"string"}
BQ_PROCESSED_DATASET = "Teste_txt2sql"  # @param {type:"string"}
MODEL_ID = "text-bison@001" # @param {type:"string"}

BUCKET_ID = "csa-datasets-public"  # @param {type:"string"}
FILENAME = "SQL_Generator_Example_Queries.csv"  # @param {type:"string"}
client = bigquery.Client(project=BQ_PROJECT_ID)
BQ_MAX_BYTES_BILLED = pow(2, 30)  # 1GB

model = TextGenerationModel.from_pretrained(MODEL_ID)
table_name = 'sales'


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



df = pd.read_csv("gs://demo_rag_t2sql_helio/salesinfo.csv", header=0)

train_df = df.loc[df["Dataset"] == "Train", ["Question", "SQL Query"]]
eval_df = df.loc[df["Dataset"] == "Eval", ["Question", "SQL Query"]]

few_examples = ""
for index, row in train_df.iterrows():
    few_examples += one_shot_template.format(
        question=row["Question"], query=row["SQL Query"]
    )




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
            if response:
                st.write("Sua resposta:")
                st.write(response.to_dataframe())
        with second_tab2:
            st.text(query)
