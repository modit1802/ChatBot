import pandas as pd
from langchain.schema import Document

def load_csv_to_documents(file_path):
    df = pd.read_csv(file_path)

    documents = []
    for index, row in df.iterrows():
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        documents.append(Document(page_content=content))

    return documents
