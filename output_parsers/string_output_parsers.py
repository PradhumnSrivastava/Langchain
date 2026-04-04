from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
load_dotenv()

llm = ChatGroq(model= "llama3-70b-8192", temperature=0.9,max_tokens=2000)

prompt = ChatPromptTemplate.from_messages([
    ("user", "What is the capital of France?")
])

chain = prompt | llm
chain.invoke({})

