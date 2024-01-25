import os
import time
import shutil
import logging
import argparse
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_core.documents.base import Document
from langchain.document_loaders    import PyPDFLoader
from langchain.embeddings          import HuggingFaceEmbeddings
from langchain.text_splitter       import CharacterTextSplitter
from langchain.vectorstores        import Chroma

load_dotenv(verbose=True)
env_model_embeddings = os.environ['MODEL_EMBEDDINGS']
env_regenerate_vs    = True if os.environ['REGENERATE_VECTORSTORE'] == "True" else False

env_cache_dir        = os.environ['CACHE_DIR']
env_log_level        = {'NOTSET':0, 'DEBUG':10, 'INFO':20, 'WARNING':30, 'ERROR':40, 'CRITICAL':50}.get(os.environ['LOG_LEVEL'], 20)

logger = logging.getLogger('Logger')
logger.addHandler(logging.StreamHandler())
logger.setLevel(env_log_level)


def read_pdf(pdf_path:str):
    loader = PyPDFLoader(pdf_path)
    pdf_pages = loader.load_and_split()
    return pdf_pages

# Split the texts (document) into smaller chunks
def split_text(pdf_pages, chunk_size=300, chunk_overlap=50, separator=''):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator)
    pdf_doc = text_splitter.split_documents(pdf_pages)
    return pdf_doc

def generate_vectorstore_from_documents(
        splitted_docs    :list[Document],
        vectorstore_path :str  = './vectorstore',
        embeddings_model :str  = 'sentence-transformers/all-mpnet-base-v2',
        normalize_emb    :bool = False,
    ) -> None:
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model, model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':normalize_emb})
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
    for doc in tqdm(splitted_docs):
        vectorstore.add_documents([doc])


def generate_vectorstore_from_pdf(pdf_path, vectorstore_path, model_embeddings, chunk_size=500, chunk_overlap=50, normalize_emb=False, regenerate=False):
    if regenerate and os.path.exists(vectorstore_path):
        shutil.rmtree(vectorstore_path)
        logger.info(f'The vectorstore "{vectorstore_path}" is deleted.')
    if not os.path.exists(vectorstore_path):
        stime = time.time()

        logger.info(f'*** Reading the document ({pdf_path})')
        pdf = read_pdf(pdf_path)
        for doc in pdf:
            doc.page_content = doc.page_content.replace('\n', ' ')  # remove all line-feed code and concatenate lines in the chunk.
            logger.debug(doc.page_content)
        logger.info(f'{len(pdf)} pages read.')

        logger.info(f'*** Splitting the document into chunks')
        logger.info(f'Chunk size={chunk_size}, Chunk overlap={chunk_overlap}')
        docs = split_text(pdf, chunk_size, chunk_overlap)
        logger.info(f'The document was splitted into {len(docs)} chunks.')

        logger.info(f'*** Generating embeddings and registering it to the vectorstore ({vectorstore_path})')
        generate_vectorstore_from_documents(docs, vectorstore_path, model_embeddings, normalize_emb=normalize_emb)
        etime = time.time()
        logger.info(f'The vectorstore generation took {etime-stime:6.2f} sec')


def parse_arguments():
    parser = argparse.ArgumentParser('vectorstore_generator', 'Generates a vectorstore from a PDF file.')
    parser.add_argument('-i', '--input_document', required=True, default=None, type=str, help='Input PDF file name')
    parser.add_argument('-o', '--output_vectorstore', default=None, type=str, help='Output vectorstore file name')
    parser.add_argument('-s', '--chunk_size', default=300, type=int, help='Chunk size')
    parser.add_argument('-v', '--chunk_overlap', default=0, type=int, help='Chunk overlap')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    if args.output_vectorstore is None:
        pdf_base_file_name = os.path.splitext(os.path.split(args.input_document)[-1])[0]
        vectorstore_path = f'vectorstore_{pdf_base_file_name}'
    else:
        vectorstore_path = args.output_vectorstore

    generate_vectorstore_from_pdf(
        pdf_path         = args.input_document, 
        vectorstore_path = vectorstore_path,
        model_embeddings = env_model_embeddings,
        chunk_size       = args.chunk_size,
        chunk_overlap    = args.chunk_overlap,
        normalize_emb    = False,
        regenerate       = env_regenerate_vs        # True: Remove the vectorstore and regenerate it if it exists
    )

if __name__ == '__main__':
    main()
