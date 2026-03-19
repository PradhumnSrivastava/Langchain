from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.7,
    max_new_tokens=200,
)

model = ChatHuggingFace(llm=llm)

print("Start chatting with the model (type 'exit' to stop):")

message = []

while True:
    prompt = input("Me: ")

    if prompt.lower() == "exit":
        print("Exiting the chat. Goodbye!")
        break

    # clearly mark roles
    message.append(f"User: {prompt}")

    # convert to single string
    full_prompt = "\n".join(message)

    response = model.invoke(full_prompt)

    # store AI response properly
    message.append(f"AI: {response.content}")

    print("Model:", response.content)