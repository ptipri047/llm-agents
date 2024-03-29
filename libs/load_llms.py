import pandas as pd
from huggingface_hub import hf_hub_download

# from langchain import LlamaCpp
from langchain_community.llms import LlamaCpp
from langchain_community.llms import Ollama


class LoadLLMS:
    def __init__(self):
        super().__init__()

    def load_ollama(self, model):
        llm = Ollama(model=model)
        return llm

    def load_huggingface_gguf_llamamodel(
        self,
        model_repoid,
        model_file,
        localdir="./cache",
        gpu_nblayers=0,
        temperature=0,
        n_batch=5,
    ):
        # downloading llm gguf image from huggingface
        # print(f'\n\ndownloading model file : {model_file}') ff
        model_path = hf_hub_download(
            model_repoid, filename=model_file, local_dir="./cache"
        )
        print("download done\n\n")

        # instantiating the LLM
        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=gpu_nblayers,
            n_batch=n_batch,
            temperature=temperature,
            verbose=False,
        )

        return llm

    def get_modelinfo_from_panda(self, size=None, archi=None):
        print("entered")

        modelsDS = pd.read_csv("./models.csv")

        if size is not None:
            modelsDS = modelsDS[modelsDS["size"] == size]

        if archi is not None:
            modelsDS = modelsDS[modelsDS["archi"] == archi]
        return modelsDS


if __name__ == "__main__":
    model_repo = "emir12/tiny_llama-1.1b-v2.gguf"
    model_file = "tiny_llama-1.1b-v2.gguf"
    myllm = LoadLLMS().load_huggingface_gguf_llamamodel(model_repo, model_file)

    print(myllm("what are the colors starting by letter b in english"))
