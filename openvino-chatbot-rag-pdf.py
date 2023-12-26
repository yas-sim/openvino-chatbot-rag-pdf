import os
import time
from dotenv import load_dotenv
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
from transformers.generation.streamers import BaseStreamer, TextStreamer

from langchain_core.documents.base import Document
from langchain.document_loaders    import PyPDFLoader
from langchain.embeddings          import HuggingFaceEmbeddings
from langchain.text_splitter       import CharacterTextSplitter
from langchain.vectorstores        import Chroma

from optimum.intel.openvino import OVModelForCausalLM

load_dotenv(verbose=True)
env_source_document  = os.environ['SOURCE_DOCUMENT']
env_chunk_size       = int(os.environ['CHUNK_SIZE'])
env_chunk_overlap    = int(os.environ['CHUNK_OVERLAP'])
env_model_embeddings = os.environ['MODEL_EMBEDDINGS']
env_vectorstore_path = os.environ['VECTORSTORE_PATH']

env_cache_dir        = os.environ['CACHE_DIR']
env_model_vendor     = os.environ['MODEL_VENDOR']
env_model_name       = os.environ['MODEL_NAME']
env_model_precision  = os.environ['MODEL_PRECISION']
env_inference_device = os.environ['INFERENCE_DEVICE']
ov_config            = {"PERFORMANCE_HINT":"LATENCY", "NUM_STREAMS":"1", "CACHE_DIR":env_cache_dir}

env_num_max_tokens   = int(os.environ['NUM_MAX_TOKENS'])
env_rag_chain_type   = os.environ['RAG_CHAIN_TYPE']
streaming=True if os.environ['STREAMING_OUTPUT'] == "True" else False


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

    del vectorstore
    del embeddings

if not os.path.exists(env_vectorstore_path):
    stime = time.time()
    print(f'*** Reading the document ({env_source_document})...', end='', flush=True)
    pdf = read_pdf(env_source_document)
    print(f'{len(pdf)} pages read')
    print(f'*** Splitting the document into chunks')
    docs = split_text(pdf, env_chunk_size, env_chunk_overlap)
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

from langchain.prompts import ChatPromptTemplate
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model_id = f'{env_model_vendor}/{env_model_name}'
ov_model_path = f'./{env_model_name}/{env_model_precision}'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=env_cache_dir)
model = OVModelForCausalLM.from_pretrained(model_id=ov_model_path, device=env_inference_device, ov_config=ov_config, cache_dir=env_cache_dir)

def generate_rag_prompt(question, vectorstore, bos_token='<s>', verbose=False):
    B_INST, E_INST = '[INST]', '[/INST]'
    B_SYS, E_SYS = '<<SYS>>\n', '\n<</SYS>>\n\n'
    reference_documents = vectorstore.similarity_search(question, k=4)
    prompt  = f'{bos_token}{B_INST} {B_SYS}You are responding to highly technical customers. '
    prompt += 'Answer the question based only on the following context:\n'
    for ref_doc in reference_documents:
        prompt += ref_doc.page_content.replace('\n', '')
    prompt += f'{E_SYS}'
    prompt += f'Question: {question} {E_INST}'
    if verbose:
        print(prompt)
    return prompt

# This class is almost identical to TextStreamer class (I implemented this just for my study purpose)
class MyStreamer(BaseStreamer):
    def __init__(self, tokenizer, **decode_kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.token_cache = []
        self.string = ''
        self.decode_kwargs = decode_kwargs
        self.print_pos = 0
    
    def put(self, value):
        if len(value.shape)>1: # Ignore the input prompt message
            return
        value = value.tolist()
        self.token_cache.extend(value)
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        if text.endswith('\n'):
            print_text = text[self.print_pos: ]
            print(print_text.rstrip('\n'))
            self.token_cache= []
            self.print_pos = 0
        else:
            print_text = text[self.print_pos: text.rfind(' ')+1].rstrip('</s>')
            self.print_pos += len(print_text)
            print(print_text, end='', flush=True)

    def end(self):
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            print_text = text[self.print_pos: ].rstrip('</s>')
            print(print_text)
            self.token_cache = []
            self.print_pos = 0
        else:
            print()
            self.token_cache = []
            self.print_pos = 0


def run_llm_text_generation(model, prompt, tokenizer, max_new_tokens=140, temperature=0.5, repetition_penalty=1.0, streaming=False):
    tokenizer_kwargs = {"add_special_tokens": False}
    tokens = tokenizer(prompt, return_tensors='pt', **tokenizer_kwargs)
    do_sample = True if temperature > 0 else False
    if streaming:
        #streamer = TextStreamer(tokenizer, skip_prompt=True, **tokenizer_kwargs)
        streamer = MyStreamer(tokenizer, **tokenizer_kwargs)
    else:
        streamer = None

    answer_tokens = model.generate(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
        max_new_tokens=max_new_tokens,
        #pad_token_id=tokenizer.pad_token_id,
        pad_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
        temperature=temperature,
        do_sample=do_sample,
        streamer=streamer,
        repetition_penalty=repetition_penalty,
    )
    answer_string = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)
    answer = answer_string[0]
    if '\nAnswer: ' in answer:
        answer = answer.split('\nAnswer: ')[1]
    return answer

while True:
    print('Question: ', end='', flush=True)
    question = input()
    prompt = generate_rag_prompt(question, vectorstore, tokenizer.bos_token)
    answer = run_llm_text_generation(model, prompt, tokenizer, streaming=streaming, temperature=0.1, repetition_penalty=1.2)
    if streaming:
        print()
    else:
        print(f'\nAnswer: {answer}')
