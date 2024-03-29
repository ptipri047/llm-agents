import streamlit as st
import os

import warnings

warnings.filterwarnings("ignore")

from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import ConversationChain


from transformers import AutoTokenizer
from ctransformers import AutoModelForCausalLM

"""
    Importing custom libraries
"""
from libs.load_llms import LoadLLMS
from libs.query_llms import QueryLLMs


"""
   Load envionment variables from .env file
"""
from dotenv import load_dotenv

load_dotenv()
print("\n\n")
print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))


# create loader
loader = LoadLLMS()
models = loader.get_modelinfo_from_panda(archi="ollama")

repoid, modelfile = models.loc[:, ["repoid", "modelfile"]].iloc[0]
myllm = loader.load_ollama(repoid)


# Run a prompt
queryObj = QueryLLMs(myllm)
queryObj.getColorsForFlowerType("Tulipe")


# agent
from langchain.agents import load_tools, initialize_agent

tools = load_tools(["serpapi", "llm-math"], myllm)


os.abort()


## Imports
from huggingface_hub import hf_hub_download
from llama_cpp import Llama


## Download the GGUF model
model_name = "aladar/tiny-random-LlamaForCausalLM-GGUF"
model_file = "tiny-random-LlamaForCausalLM.gguf"
model_path = hf_hub_download(model_name, filename=model_file, local_dir="./cache")

print(f"modelpath:{model_path}")

"""
## Instantiate model from downloaded file
llm = Llama(
    model_path=model_path,
    n_ctx=16000,  # Context length to use
    n_threads=32,            # Number of CPU threads to use
    n_gpu_layers=0        # Number of model layers to offload to GPU
)

## Generation kwargs
generation_kwargs = {
    "max_tokens":20000,
    "stop":["</s>"],
    "echo":False, # Echo the prompt in the output
    "top_k":1 # This is essentially greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
}

## Run inference
prompt = "The meaning of life is "
res = llm(prompt, **generation_kwargs) # Res is a dictionary

## Unpack and the generated text from the LLM response dictionary and print it
print(res["choices"][0]["text"])
# res is short for result

"""

# model="aladar/tiny-random-LlamaForCausalLM-GGUF"
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={"temperature": 0.7})


## Function To get response from LLAma 2 model


def getLLamaresponse(input_text, no_words, blog_style):
    ### LLama2 model
    llm = CTransformers(
        model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        config={"max_new_tokens": 256, "temperature": 0.01},
    )

    ## Prompt Template
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
            """

    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"], template=template
    )

    ## Generate the ressponse from the LLama 2 model
    response = llm(
        prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
    )
    print(response)
    return response


st.set_page_config(
    page_title="Generate Blogs",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

## creating to more columns for additonal 2 fields

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input("No of Words")
with col2:
    blog_style = st.selectbox(
        "Writing the blog for",
        ("Researchers", "Data Scientist", "Common People"),
        index=0,
    )

submit = st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))
