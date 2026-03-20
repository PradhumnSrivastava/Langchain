from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                          temperature=0.9,
                          max_new_tokens=100
)

model = ChatHuggingFace(llm=llm)

template = ChatPromptTemplate.from_messages([
    ("human", "Explain {input} in simple terms.")
])

#message = []

while True:
    input_text = input("Enter a topic to explain (or 'exit' to quit): ")
    
    if input_text.lower() == 'exit':
        break
    
    messages = template.format_messages(input=input_text)

    response = model.invoke(messages)
    print(f"AI: {response.content}")