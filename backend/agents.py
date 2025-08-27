import os
from crewai import Agent, Task, Flow, Crew
from langchain_groq import ChatGroq
from tools import kb_lookup_tool
from dotenv import load_dotenv
from crewai.llms import LLM


load_dotenv()

# CrewAI native LLM wrapper (works with Groq)
llm = LLM(
    model="groq/gemma2-9b-it",   # ✅ tell CrewAI it's Groq
    api_key=os.getenv("GROQ_API_KEY")
)
intent_classifier = Agent(
    role="Senior customer support executive with specialization in intent classification and sentiment analysis.",
    goal="Classify user's issue type, urgency and sentiment based on their query that helps in routing the issue to the right support team.",
    backstory="You are en expert in customer support and can classify user queries into different intents such as 'technical issue', 'billing', 'general inquiry', etc. You also assess the urgency of the issue and the sentiment of the user's message.",
    verbose=True,
    llm=llm
)

FAQ_agent = Agent(
    role="FAQ and knowledge base specialist.",
    goal="Provide accurate answers to frequently asked questions and assist users in finding relevant information.",
    backstory="You are an expert in the company's products and services, with access to a vast knowledge base. You can answer common questions and guide users to the right resources.",
    verbose=True,
    llm=llm
)

response_generator = Agent(
    role="Communication director with expertise in effective response generation.",
    goal="Generate clear and concise responses to user inquiries based on their intent and sentiment.",
    backstory="You are a skilled communicator who can craft responses that address user concerns while aligning with company policies and tone.",
    verbose=True,
    llm=llm
)

escalation_agent = Agent(
    role="Escalation specialist.",
    goal="Determine if the query needs human intervention and escalate if necessary",
    backstory="You are an expert in issue escalation processes and can quickly determine if the user's issue can be auto-resolved or needs human support.",
    verbose=True,
    llm=llm
)
