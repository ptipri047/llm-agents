from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class QueryLLMs:
    def __init__(self,llm):
        super().__init__()
        self.llm = llm

    def getColorsForFlowerType(self, flowertype):
        prompt_template = (
            """what are the colors for this type of flowers {flowertype}"""
        )
        prompt = PromptTemplate(
            input_variables=["flowertype"], template=prompt_template
        )

        output_parser = StrOutputParser()
        chain = prompt | self.llm | output_parser
        print(chain.invoke({"flowertype": flowertype}))
