#langchain building blocks
import warnings
import os
from pathlib import Path
from dotenv import load_dotenv
warnings.filterwarnings("ignore", category=UserWarning)

# Find the .env file next to this script
script_dir = Path(__file__).resolve().parent
env_path = script_dir / ".env"

load_dotenv(dotenv_path=env_path)

# Check if Gemini API key is loaded
print(
    "GOOGLE_API_KEY found:",
    bool(os.getenv("GOOGLE_API_KEY"))
)

from langchain.chat_models import init_chat_model

# Initialize Google Gemini model
model = init_chat_model(
    "gemini-3.6-flash",
    model_provider="google_genai"
)

# Invoke the model
response = model.invoke("Hello, how are you?")

print("Model Response:")
print(response.content)


# Prompt template
# Simple prompt template
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# PromptTemplate = a single plain-text string with blanks.
simple_prompt_template = PromptTemplate(input_variables=["topic"], template="Write a short story about {topic}.")


# .format() fills the blank and returns the finished text.
formatted= simple_prompt_template.format(topic="Virat Kolhi")
print("=== Formatted Prompt ===") # Write a short story about Virat Kolh
print()

# ChatPromptTemplate = built from ROLES (system / human), which is how chat
# models expect their input. "system" sets behavior, "human" is the user turn.
from langchain_core.prompts import ChatPromptTemplate
chat_template = ChatPromptTemplate.from_messages([("system", "You are a helpful coding tutor. Keep answers short and clear."),
("human", "Explain {concept} with a simple Python example.")])

# .format_messages() fills the blanks and returns a LIST of messages.
messages = chat_template.format_messages(concept="decorators") # {concept} in the template is the empty blank, and concept="list comprehension" fills it.

print("=== Chat Messages ===")
for msg in messages:
    print(f" [{msg.type}]: {msg.content[:80]}...")
print()


# ==========================================
# 3. CHAINS – Connecting the Pieces with LCEL
# ==========================================
# The pipe | glues components into a pipeline. Each step's output flows
# into the next, left to right: prompt | model | parser.
from langchain_core.output_parsers import StrOutputParser
# prompt fills the blanks → model answers → parser pulls out clean text.
# StrOutputParser just extracts the plain string from the model's reply
# object, so you don't have to write .content yourself every time.
chain = chat_template | model | StrOutputParser()

# Run the whole pipeline with a single .invoke().
result = chain.invoke({"concept": "for each loops"})
print("=== Chain Output ===")
print(result)
print()

# ==========================================
# 4. MEMORY – Giving the Model Context
# ==========================================

# Models are stateless – they forget everything between calls.
# "Memory" is simply us storing past messages and feeding them back in.
# (Modern LangChain uses ChatMessageHistory; the old
# ConversationBufferMemory is deprecated and out of the core package.)

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# A simple in-memory store that holds the conversation.
memory = InMemoryChatMessageHistory()
# Hand-build a short conversation so we have something to "remember".
memory.add_message(HumanMessage(content="Hello, who are you?"))
memory.add_message(AIMessage(content="Hello, I am fine"))
memory.add_message(
    HumanMessage(content="My name is Alex and I'm learning LangChain")
)
memory.add_message(
    AIMessage(content="Nice to meet you, Alex! LangChain is a great choice.")
)
print("=== Memory Contents ===")
for msg in memory.messages:
    print(f"  [{msg.type}]: {msg.content[:80]}...")
print()

# The "placeholder" slot is where the stored messages get injected into the
# prompt, so the model can see the earlier conversation.

chat_with_memory = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful tutor. Use the conversation history to personalize your responses."
    ),
    (
        "placeholder",
        "{history}"
    ),  # past messages get dropped in here
    (
        "human",
        "{question}"
    ),
])

chain_with_memory = chat_with_memory | model | StrOutputParser()

# We pass the stored history in alongside the new question.
# Watch the model correctly recall the name "Alex" – that's "memory".

result = chain_with_memory.invoke({
    "history": memory.messages,
    "question": "What was my name again?"
})

print("=== Memory-Aware Response ===")
print(result)
print()