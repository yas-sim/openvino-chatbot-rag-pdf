import os
import time
import logging
import argparse
from dotenv import load_dotenv

from transformers import AutoModel, AutoTokenizer
from transformers.generation.streamers import TextStreamer
from langchain.embeddings          import HuggingFaceEmbeddings
from langchain.vectorstores        import Chroma

from optimum.intel.openvino import OVModelForCausalLM

load_dotenv(verbose=True)
env_model_embeddings = os.environ['MODEL_EMBEDDINGS']

env_cache_dir        = os.environ['CACHE_DIR']
env_model_vendor     = os.environ['MODEL_VENDOR']
env_model_name       = os.environ['MODEL_NAME']
env_model_precision  = os.environ['MODEL_PRECISION']
env_inference_device = os.environ['INFERENCE_DEVICE']
ov_config            = {"PERFORMANCE_HINT":"LATENCY", "NUM_STREAMS":"1", "CACHE_DIR":env_cache_dir}

env_streaming        = True if os.environ['STREAMING_OUTPUT'] == "True" else False

env_log_level        = {'NOTSET':0, 'DEBUG':10, 'INFO':20, 'WARNING':30, 'ERROR':40, 'CRITICAL':50}.get(os.environ['LOG_LEVEL'], 20)

logger = logging.getLogger('Logger')
logger.addHandler(logging.StreamHandler())
logger.setLevel(env_log_level)

def generate_rag_prompt(question, vectorstore, bos_token='<s>', verbose=False):
    B_INST, E_INST = '[INST]', '[/INST]'
    B_SYS, E_SYS = '<<SYS>>\n', '\n<</SYS>>\n\n'
    reference_documents = vectorstore.similarity_search(question, k=4)
    prompt  = f'{bos_token}{B_INST} {B_SYS}You are responding to highly technical customers. '
    prompt += 'Answer the question based only on the following context:\n'
    for ref_doc in reference_documents:
        prompt += ref_doc.page_content.replace('\n', '') + '\n'
        prompt += '\n'
    prompt += f'{E_SYS}'
    prompt += f'Question: {question} {E_INST}'
    logger.debug(prompt)
    if verbose:
        print(prompt)
    return prompt

def generate_rag_prompt_jp(question, vectorstore, bos_token='<s>', verbose=False):
    B_INST, E_INST = '', '\n'
    B_SYS, E_SYS = '設定: ', '\n\n'
    reference_documents = vectorstore.similarity_search(question, k=4)
    prompt  = f'{bos_token}{B_INST} {B_SYS} 以下の参考文章のみを参照して質問に回答しなさい。\n'
    for ref_doc in reference_documents:
        prompt += ref_doc.page_content.replace('\n', ' ')
        prompt += '\n'
    prompt += f'{E_SYS}'
    prompt += f'ユーザー: {question} {E_INST}'
    prompt += 'システム: '
    logger.debug(prompt)
    if verbose:
        print(prompt)
    return prompt

def run_llm_text_generation(model, prompt, tokenizer, max_new_tokens=140, temperature=0.5, repetition_penalty=1.0, streaming=False, verbose=False):
    tokenizer_kwargs = {"add_special_tokens": False}
    tokens = tokenizer(prompt, return_tensors='pt', **tokenizer_kwargs)
    do_sample = True if temperature > 0 else False
    if streaming:
        streamer = TextStreamer(tokenizer, skip_prompt=True, **tokenizer_kwargs)
    else:
        streamer = None

    answer_tokens = model.generate(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        #pad_token_id=tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,

        temperature=temperature,
        do_sample=do_sample,
        streamer=streamer,
        repetition_penalty=repetition_penalty,
    )
    answer_string = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)
    answer = answer_string[0]
    if '\nAnswer: ' in answer:
        answer = answer.split('\nAnswer: ')[1]
    if '\nシステム: ' in answer:
        answer = answer.split('\nシステム: ')[1]
    return answer


def parse_arguments():
    parser = argparse.ArgumentParser('OpenVINO LLM chat-bot with RAG', 'Chatbot answers your queries by referring a vectorstore generated from a PDF file.')
    parser.add_argument('-v', '--vectorstore', required=True, default=None, type=str, help='Path to a vectorstore')
    parser.add_argument('-q', '--query', default=None, type=str, help='Query message')
    parser.add_argument('-m', '--max_tokens', default=140, type=int, help='Number of maximum tokens to generate')
    parser.add_argument('-t', '--temperature', default=0.2, type=float, help='Temperature')
    parser.add_argument('-r', '--repetition_penalty', default=1.0, type=float, help='Repetition penalty')
    parser.add_argument('--verbose', action='store_true', help='Verbose flag')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    logger.info(f'Vectorstore: {args.vectorstore}')

    # Create a vectorstore object
    #
    ### WORKAROUND in case "trust_remote_code=True is required error" occurs in HuggingFaceEmbeddings()
    #model = AutoModel.from_pretrained(env_model_embeddings, trust_remote_code=True, cache_dir=cache_dir) 
    embeddings = HuggingFaceEmbeddings(model_name=env_model_embeddings, model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':True})
    vectorstore = Chroma(persist_directory=args.vectorstore, embedding_function=embeddings)

    model_id = f'{env_model_vendor}/{env_model_name}'
    ov_model_path = f'./{env_model_name}/{env_model_precision}'
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=env_cache_dir)
    model = OVModelForCausalLM.from_pretrained(model_id=ov_model_path, device=env_inference_device, ov_config=ov_config, cache_dir=env_cache_dir)

    logger.info(f'LLM Model: {ov_model_path}')

    while True:
        if args.query is None:
            print('Question: ', end='', flush=True)
            query = input()
            if query == '':
                break
        else:
            query = args.query
        
        if env_model_name in [ 'youri-7b-chat' ]:
            print(f'Japanese model ({env_model_name})')
            prompt = generate_rag_prompt_jp(query, vectorstore, tokenizer.bos_token, args.verbose)
        else:
            prompt = generate_rag_prompt(query, vectorstore, tokenizer.bos_token, args.verbose)

        answer = run_llm_text_generation(model, prompt, tokenizer, max_new_tokens=args.max_tokens, streaming=env_streaming, temperature=args.temperature, repetition_penalty=args.repetition_penalty, verbose=args.verbose)
        if env_streaming:
            print()
        else:
            print(f'\nAnswer: {answer}')
        if args.query is not None:
            break

if __name__ == '__main__':
    main()
