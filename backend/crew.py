from crewai.flow.flow import Flow, start, listen
from agents import intent_classifier, FAQ_agent, response_generator, escalation_agent
from tools import kb_lookup_tool
from litellm import completion

class SupportFlow(Flow):
    """Flow to resolve EarnIn support queries using agent crew."""

    @start()
    def classify_intent(self):
        user_query = self.state["user_query"]   # ✅ FIXED: access dict directly
        result = intent_classifier.kickoff(user_query)
        # Normalize agent output to string for downstream steps
        intent_text = getattr(result, "output", None) or str(result)
        self.state["intent"] = intent_text
        return intent_text

    @listen(classify_intent)
    def lookup_kb(self, prev_output):
        # prev_output can be a pydantic model; use its .output or str()
        query = getattr(prev_output, "output", None) or str(prev_output)
        kb_result = kb_lookup_tool(query)
        self.state["kb"] = kb_result
        return kb_result

    @listen(lookup_kb)
    def generate_response(self, kb_result):
        input_str = f"Intent: {self.state['intent']}\nKB Info: {kb_result}\n"
        resp = response_generator.kickoff(input_str)
        response_text = getattr(resp, "output", None) or str(resp)
        self.state["response"] = response_text
        return response_text

    @listen(generate_response)
    def finalize(self, response):
        if "escalate" in str(self.state["intent"]).lower():
            decision = escalation_agent.kickoff(response)
        else:
            decision = {"resolution": "auto-resolved", "reply": response}
        return decision


# Run the flow
if __name__ == "__main__":
    flow = SupportFlow()
    result = flow.kickoff(inputs={"user_query": "I didn't receive my payment today."})
    print(result)
