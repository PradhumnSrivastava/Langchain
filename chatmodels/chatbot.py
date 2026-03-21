from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                          temperature=0.9,
                          max_new_tokens=100
)

model = ChatHuggingFace(llm=llm)


messages = [SystemMessage(content="You are a helpful assistant that explains topics in simple terms.")
             ]

while True:
    input_text = input("Enter a topic to explain (or 'exit' to quit): ")
    
    if input_text.lower() == 'exit':
        break
    messages.append(HumanMessage(content=input_text))

    response = model.invoke(messages)
    
    messages.append(AIMessage(content=response.content))
    print(f"AI: {response.content}")
    print(messages)