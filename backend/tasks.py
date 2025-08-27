from crewai import Agent, Task, Flow, Crew

from agents import intent_classifier, FAQ_agent as kb_lookup, response_generator, escalation_agent



intent_classification_task = Task(
    description="Classify the user's support query based on issue type, urgency and sentiment.",
    expected_output="An intent label (e.g., 'Payment Delay', 'App Error') and urgency (Low/High) and sentiment (Happy/Sad/Angry etc.).",
    agent=intent_classifier
)

kb_lookup_task = Task(
    description="Search the internal knowledge base to find the most relevant solution article.",
    expected_output="A summarized solution from FAQ or documentation.",
    agent=kb_lookup
)

response_generation_task = Task(
    description="Write a personalized response to the user incorporating the found solution.",
    expected_output="A complete response message ready to be sent to the user.",
    agent=response_generator
)

escalation_task = Task(
    description="Decide whether the issue is fully resolved or needs to be escalated to a human agent.",
    expected_output="Resolution decision (Auto-Resolved/Escalated) with reasoning.",
    agent=escalation_agent
)