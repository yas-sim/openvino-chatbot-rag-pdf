import os
import time
from dotenv import load_dotenv
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, pipeline

from langchain_core.documents.base import Document
from langchain.document_loaders    import PyPDFLoader
from langchain.embeddings          import HuggingFaceEmbeddings
from langchain.text_splitter       import CharacterTextSplitter
from langchain.vectorstores        import Chroma
from langchain.llms                import HuggingFacePipeline
from langchain.chains              import RetrievalQA

from optimum.intel.openvino import OVModelForCausalLM

load_dotenv(verbose=True)
env_model_embeddings = os.environ['MODEL_EMBEDDINGS']
env_vectorstore_path = os.environ['VECTORSTORE_PATH']
env_source_document  = os.environ['SOURCE_DOCUMENT']
env_cache_dir        = os.environ['CACHE_DIR']
env_model_vendor     = os.environ['MODEL_VENDOR']
env_model_name       = os.environ['MODEL_NAME']
env_model_precision  = os.environ['MODEL_PRECISION']
env_inference_device = os.environ['INFERENCE_DEVICE']
ov_config            = {"PERFORMANCE_HINT":"LATENCY", "NUM_STREAMS":"1", "CACHE_DIR":env_cache_dir}

env_num_max_tokens   = int(os.environ['NUM_MAX_TOKENS'])
env_rag_chain_type   = os.environ['RAG_CHAIN_TYPE']


def read_pdf(pdf_path:str):
    loader = PyPDFLoader(pdf_path)
    pdf_pages = loader.load_and_split()
    return pdf_pages


def split_text(pdf_pages, chunk_size=300, chunk_overlap=50, separator=''):
    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap, 
        separator = separator
    )
    pdf_doc = text_splitter.split_documents(pdf_pages)
    return pdf_doc


def generate_vectorstore_from_documents(
        splitted_docs    :list[Document],
        vectorstore_path :str  = './vectorstore',
        embeddings_model :str  = 'sentence-transformers/all-mpnet-base-v2',
        pipeline         :str  = 'en_core_web_sm',
        normalize_emb    :bool = False,
    ) -> None:

    embeddings = HuggingFaceEmbeddings(
        model_name = embeddings_model,
        model_kwargs = {'device':'cpu'},
        encode_kwargs = {'normalize_embeddings':normalize_emb}
    )

    vectorstore = Chroma(
        persist_directory=vectorstore_path, 
        embedding_function=embeddings
    )

    for doc in tqdm(splitted_docs):
        vectorstore.add_documents([doc])



if not os.path.exists(env_vectorstore_path):
    stime = time.time()
    print(f'*** Reading the document ({env_source_document})...', end='', flush=True)
    pdf = read_pdf(env_source_document)
    print(f'{len(pdf)} pages read')
    print(f'*** Splitting the document into chunks')
    docs = split_text(pdf, 500, 50)
    print(f'*** Generating embeddings and registering it to the vectorstore')
    generate_vectorstore_from_documents(docs, env_vectorstore_path, env_model_embeddings)
    etime = time.time()
    print(f'The vectorstore generation took {etime-stime:6.2f} sec')

#
# -------------------------------------------------------------------------------
#
### WORKAROUND in case "trust_remote_code=True is required error" occurs in HuggingFaceEmbeddings()
#model = AutoModel.from_pretrained(env_model_embeddings, trust_remote_code=True, cache_dir=cache_dir) 

embeddings = HuggingFaceEmbeddings(
    model_name = env_model_embeddings,
    model_kwargs = {'device':'cpu'},
    encode_kwargs = {'normalize_embeddings':True}
)

vectorstore = Chroma(persist_directory=env_vectorstore_path, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
#retriever = vectorstore.as_retriever(
#    search_type= 'similarity_score_threshold',
#    search_kwargs={
#        'score_threshold' : 0.5, 
#        'k' : 4
#    }
#)
#retriever = vectorstore.as_retriever(
#    search_type= 'mmr'
#)

model_id = f'{env_model_vendor}/{env_model_name}'
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=env_cache_dir)
ov_model_path = f'./{env_model_name}/{env_model_precision}'
model = OVModelForCausalLM.from_pretrained(model_id=ov_model_path, device=env_inference_device, ov_config=ov_config, cache_dir=env_cache_dir)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=env_num_max_tokens)
llm = HuggingFacePipeline(pipeline=pipe)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type=env_rag_chain_type, retriever=retriever)
#qa_chain.run('warm up')

while True:
    print('\n\n', '-' * 60)
    print(f'\nQuestion: ', end='', flush=True)
    question = input()
    if question == '':
        break
    stime = time.time()
    ans = qa_chain.run(question)
    etime = time.time()

    print(f'Answer: {ans}')
    print(f'{etime-stime:6.2f} sec')
