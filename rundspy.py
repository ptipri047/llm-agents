import sys
import os
import logging
import time

# proxy.py
import proxy
from libs.waitproxyplugin import SleepInRequests

import requests
from enum import Enum

from dotenv import load_dotenv

# dspy
import dspy
from dspy.datasets import HotPotQA
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate

logging.disable(logging.WARNING)

gsmquestion = """
Tobias is buying a new pair of shoes that costs $95. 
He has been saving up his money each month for the past three months. 
He gets a $5 allowance a month. He also mows lawns and shovels driveways. 
He charges $15 to mow a lawn and $7 to shovel. 
After buying the shoes, he has $15 in change. If he mows 4 lawns, how many driveways did he shovel?
"""  # answer = 110

REPO_PATH = "./dspylogs"
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

# Set up the cache for this notebook
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(REPO_PATH, "cache")


# import pkg_resources  # Install the package if it's not installed
class dataset_type_name(Enum):
    HOTPOTQA = 1
    GSM8K = 2


class Dspy_Test:
    def __init__(
        self,
        datasettype: dataset_type_name,
        realdataset: bool = True,
        model="meta-llama/Llama-2-13b-hf",
    ):
        self.datasettype = datasettype

        if datasettype == dataset_type_name.HOTPOTQA:
            self.load_dataset_hotpotqa(realdataset)
        else:
            self.load_dataset_qsm8k(realdataset)

        self.loadingenv()
        self.get_signature()
        self.get_modules()

        self.get_together_llm(model=model)
        self.get_retrieval_store()
        self.configure_env()

    def loadingenv(self):
        """Load envionment variables from .env file"""
        load_dotenv()
        print(f'base-{os.getenv("TOGETHER_API_BASE")}-')
        print(f'key-{os.getenv("TOGETHER_API_KEY")}-')

    def get_together_llm(self, model="meta-llama/Llama-2-13b-hf"):
        # dspy/docs/docs/deep-dive/language_model_clients
        # dsp/modules
        tog = dspy.Together(model=model)
        
        import requests
        import functools
        s = tog.session
        s.request = functools.partial(s.request, timeout=3000)
        dspy.configure(lm=tog)
        self.llm = tog

    def get_retrieval_store(self):
        colbertv2_wiki17_abstracts = dspy.ColBERTv2(
            url="http://20.102.90.50:2017/wiki17_abstracts"
        )
        self.store = colbertv2_wiki17_abstracts

    def configure_env(self, model=None):
        # self.loadingenv()
        # self.get_together_llm(model=model)
        # self.get_retrieval_store()
        dspy.settings.configure(lm=self.llm, rm=self.store)

    def load_dataset_qsm8k(self, realdataset):
        gms8k = GSM8K()
        gsm8k_trainset, gsm8k_devset = gms8k.train[:10], gms8k.dev[:10]
        self.dataset = {"gsmtrain": gsm8k_trainset, "gsmdev": gsm8k_devset}

    def load_dataset_hotpotqa(self, realdataset: bool):
        """
        *** load dataset HotPotQA
        HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable .
        """
        # Load the dataset.
        if not realdataset:
            self.load_dummy_dataset()
            return

        dataset = HotPotQA(
            train_seed=1, train_size=10, eval_seed=2023, dev_size=50, test_size=0
        )

        # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
        trainset = [x.with_inputs("question") for x in dataset.train]
        devset = [x.with_inputs("question") for x in dataset.dev]

        print(f"loading hotpotqa dataset. train:{len(trainset)}, dev:{len(devset)}")

        train_example = trainset[0]
        print(f"Question: {train_example.question}")
        print(f"Answer: {train_example.answer}")

        dev_example = devset[18]
        print(f"Question: {dev_example.question}")
        print(f"Answer: {dev_example.answer}")
        print(f"Relevant Wikipedia Titles: {dev_example.gold_titles}")

        self.dataset = {"train": trainset, "dev": devset, "dev_example": dev_example}

    def load_dummy_datasethotpotqa(self):
        class tt:
            def __init__(self, question):
                self.question = question

        dev_example = tt(
            "What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?"
        )

        self.dataset = {"train": None, "dev": None, "dev_example": dev_example}

    def get_signature(self):
        """
        **** provide signature
        """

        class BasicQA(dspy.Signature):
            """Answer questions with short factoid answers."""

            question = dspy.InputField()
            answer = dspy.OutputField(desc="often between 1 and 5 words")

        class GenerateAnswer(dspy.Signature):
            """Answer questions with short factoid answers."""

            context = dspy.InputField(desc="may contain relevant facts")
            question = dspy.InputField()
            answer = dspy.OutputField(desc="often between 1 and 5 words")

        self.signature = {"basic": BasicQA, "generate": GenerateAnswer}

    def get_modules(self):
        # for hotpotqa
        sig = self.signature["generate"]

        class RAG(dspy.Module):
            def __init__(self, num_passages=3):
                super().__init__()

                self.retrieve = dspy.Retrieve(k=num_passages)
                self.generate_answer = dspy.ChainOfThought(sig)

            def forward(self, question):
                context = self.retrieve(question).passages
                time.sleep(4)
                prediction = self.generate_answer(context=context, question=question)
                return dspy.Prediction(context=context, answer=prediction.answer)

        # for gsm 8 k
        class CoT(dspy.Module):
            def __init__(self):
                super().__init__()
                self.prog = dspy.ChainOfThought("question -> answer")

            def forward(self, question):
                return self.prog(question=question)

        self.modules = {"rag": RAG, "CoT": CoT}

    def call_predict_1(self):
        """
        *** calling with predict
        """

        print("\n\n****************** normal predict*******************************")
        # Define the predictor.
        generate_answer = dspy.Predict(self.signature["basic"])

        # Call the predictor on a particular input.
        pred = generate_answer(question=self.dataset["dev_example"].question)

        print(f"Predicted Answer: {pred.answer}")

        print("\n\n###### inspect history")
        self.llm.inspect_history(n=1)

    def call_cos_1(self):
        """
        Using chain of thoughts
        """

        print("\n\n************************* chain of thoughts **********************")
        lhb = len(self.llm.history)
        generate_answer_with_chain_of_thought = dspy.ChainOfThought(
            self.signature["basic"]
        )

        # Call the predictor on the same input.
        pred = generate_answer_with_chain_of_thought(
            question=self.dataset["dev_example"].question
        )

        # print(f"Thought: {pred.rationale.split('.', 1)[1].strip()}")
        print(f"Thought (cos): {pred.rationale.split('.', 1)}")
        # print(f"Thought: {pred.rationale}")
        print(f"\nPredicted Answer (cos): {pred.answer}")

        print("\n\n###### inspect history (cos)")
        lhe = len(self.llm.history)
        self.llm.inspect_history(n=lhe - lhb)

    def call_from_datastore_1(self):
        devex = self.dataset["dev_example"]
        train = self.dataset["train"]
        ragmodule = self.modules["rag"]

        """
        Retrieve from datastore
        """

        print("\n\n*********retrieve from vector store************")
        retrieve = dspy.Retrieve(k=3)
        topK_passages = retrieve(devex.question).passages

        print("\n#### data from vector store")

        print(
            f"Top {retrieve.k} passages for question: {devex.question} \n",
            "-" * 30,
            "\n",
        )

        for idx, passage in enumerate(topK_passages):
            print(f"{idx+1}]", passage, "\n")

        print("\n####through LLM")

        # Validation logic: check that the predicted answer is correct.
        # Also check that the retrieved context does actually contain that answer.
        def validate_context_and_answer(example, pred, trace=None):
            answer_EM = dspy.evaluate.answer_exact_match(example, pred)
            answer_PM = dspy.evaluate.answer_passage_match(example, pred)
            return answer_EM and answer_PM

        # Set up a basic teleprompter, which will compile our RAG program.
        teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

        # Compile!
        compiled_rag = teleprompter.compile(ragmodule(), trainset=train)

        # Ask any question you like to this simple RAG program.
        my_question = "What castle did David Gregory inherit?"

        # Get the prediction. This contains `pred.context` and `pred.answer`.
        pred = compiled_rag(my_question)

        # Print the contexts and the answer.
        print(f"Question: {my_question}")
        print(f"Predicted Answer: {pred.answer}")
        print(
            f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}"
        )

        print("\n\n### inspect history")
        self.llm.inspect_history(n=1)

    def call_gsm_2(self, customrequest=""):
        print("\n\n***********running call_gsm_2")
        # Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
        config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

        # Optimize! Use the `gms8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
        teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
        module = self.modules["CoT"]
        optimized_cot = teleprompter.compile(
            module(), trainset=self.dataset["gsmtrain"], valset=self.dataset["gsmdev"]
        )

        print("\n\n***********now evaluate call_gsm_2")

        # evaluate
        # Set up the evaluator, which can be used multiple times.
        evaluate = Evaluate(
            devset=self.dataset["gsmdev"],
            metric=gsm8k_metric,
            num_threads=4,
            display_progress=True,
            display_table=0,
        )

        # Evaluate our `optimized_cot` program.
        evaluate(optimized_cot)

        self.llm.inspect_history(n=1)

        # free time
        optimized_cot(question=customrequest)

    def call_google(self,customrequest=''):
        pass

    def test_call_hotpotqa_1(modelname):
        myDspy = Dspy_Test(
            realdataset=True, datasettype=dataset_type_name.HOTPOTQA, model=modelname
        )
        myDspy.call_predict_1()
        myDspy.call_cos_1()
        myDspy.call_from_datastore_1()

    def test_call_gsm_2(modelname):
        myDspy = Dspy_Test(
            realdataset=True,
            datasettype=dataset_type_name.GSM8K,
            model=modelname,
        )
        myDspy.call_gsm_2(customrequest = gsmquestion)

    def test_call_google(modelname, googlerequest):
        myDspy = Dspy_Test(
            realdataset=True,
            datasettype=dataset_type_name.GSM8K,
            model=modelname,
        )
        myDspy.call_google(customrequest = googlerequest)


RUN_TYPES = {
    "hotpotqa": {"funct": Dspy_Test.test_call_hotpotqa_1, "arg": ["model"]},
    "gsm8k": {"funct": Dspy_Test.test_call_gsm_2, "arg": ["model"]},
    "google_search": {"funct": Dspy_Test.test_call_google, "arg": ["model", "googlerequest"]},
}


if __name__ == "__main__":
    with proxy.Proxy(
        [
            "--sleeptime",
            "2",
            "--num-acceptors",
            "1",
            "--num-workers",
            "1",
            "--log-level",
            "d",
        ],
        plugins=[SleepInRequests],
    ):
        # run variables
        runtype = "google_search"
        model = "meta-llama/Llama-2-13b-hf"
        googlerequest = "wwho win the last wimbledon"
        
        # eval arguments
        currentrun = RUN_TYPES[runtype]
        funct = currentrun['funct']
        arguments= currentrun['arg']
        ar = [eval(z) for z in arguments]

        # run the function 
        funct(*ar)
        print("there")



'''
       dspy.configure(lm=dspy.Clarifai(model=MODEL_URL,
                                        api_key=CLARIFAI_PAT,
                                        inference_params={"max_tokens":100,'temperature':0.6}))'''