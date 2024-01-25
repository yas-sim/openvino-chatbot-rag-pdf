import os
import argparse
from dotenv import load_dotenv

from langchain.embeddings          import HuggingFaceEmbeddings
from langchain.vectorstores        import Chroma

parser = argparse.ArgumentParser('vectorstore_generator', 'Generates a vectorstore from a PDF file.')
parser.add_argument('-v', '--vectorstore', required=True, default=None, type=str, help='Path to a vectorstore')
args = parser.parse_args()

load_dotenv(verbose=True)
env_model_embeddings = os.environ['MODEL_EMBEDDINGS']

embeddings = HuggingFaceEmbeddings(model_name=env_model_embeddings, model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':True})
vectorstore = Chroma(persist_directory=args.vectorstore, embedding_function=embeddings)

while True:
    print('Query: ', end='', flush=True)
    question = input()
    if question == '':
        break
    reference_documents = vectorstore.similarity_search(question, k=4)
    for n, doc in enumerate(reference_documents):
        print('\n', n, doc.metadata, doc.page_content)
