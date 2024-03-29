import random
from typing import List, Optional, Union

import dsp
from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction
import requests


class SearchGoogle(Parameter):
    name = "SearchGoogle"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant passages from a corpus, by performing a google search"

    def __init__(self, k=3):
        self.k = k

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> Prediction:
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [query.strip().split('\n')[0].strip() for query in queries]

        # print(queries)
        # TODO: Consider removing any quote-like markers that surround the query too.
        k = k if k is not None else self.k
        
        # https://www.google.com/search?q=judo+demain

        passages = dsp.retrieveEnsemble(queries, k=k)
        return Prediction(passages=passages)