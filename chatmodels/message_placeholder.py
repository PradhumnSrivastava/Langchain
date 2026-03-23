from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# Step 1: chat_history load karo
chat_history = []

with open("chat_history.txt", "r") as f:
    for line in f:
        if line.startswith("Human:"):
            chat_history.append(HumanMessage(content=line.replace("Human:", "").strip()))
        elif line.startswith("AI:"):
            chat_history.append(AIMessage(content=line.replace("AI:", "").strip()))

# Step 2: user input
user_query = input("Type your query: ")

# Step 3: invoke
prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": user_query
})

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    max_new_tokens=50
)

model = ChatHuggingFace(llm=llm)

response = model.invoke(prompt)

print("\nAI Response:")
print(response.content)
