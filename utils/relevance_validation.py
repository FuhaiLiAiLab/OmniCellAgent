import re

# State dict for managing relevance checking
state = {
    "query": None,
    "tool_responses": {},
    "relevance_checked": False,
    "ranking": [],
    "irrelevant_responses": [],
    "reasoning": {}
}

# Single-pass relevance checking function for multiple tool responses (optimized for testing)
def check_relevance(client, query: str, tool_responses: dict, previous_reasoning: str = None, state={}, model: str = "gpt-4o", relevance_threshold: int = 5):
    prompt = (
        "Evaluate the relevance of the following tool responses to the user's query."
        " For each tool, clearly state 'Relevant' or 'Irrelevant', followed by a relevance score in the exact format '[Score: X]',"
        " where X is an integer from 1 (lowest) to 10 (highest). Briefly justify each answer."
        " Finally, rank the responses from most to least relevant."
        f"\n\nUser's Query: '{query}'"
    )

    for tool_name, response_text in tool_responses.items():
        prompt += f"\n\nTool '{tool_name}' Response: '{response_text}'"

    if previous_reasoning:
        prompt += f"\n\nPrevious Reasoning: '{previous_reasoning}'"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You evaluate and rank the relevance of tool outputs to user queries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    result = response.choices[0].message.content.strip()
    state["reasoning"]["full_response"] = result
    
    relevance_pattern = r"Tool '(.*?)' Response:.*?(Relevant|Irrelevant).*?\[Score: (\d+)\]"
    matches = re.findall(relevance_pattern, result, re.DOTALL)

    relevance_scores = {}
    for tool, relevance, score in matches:
        score = int(score)
        relevance_scores[tool] = score
        state["reasoning"][tool] = {"relevance": relevance, "score": score}

    ranked_tools = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)

    state["ranking"] = [tool for tool, _ in ranked_tools]
    state["irrelevant_responses"] = [tool for tool, score in relevance_scores.items() if score < relevance_threshold]
    state["relevance_checked"] = True

    return {
        "ranking": state["ranking"],
        "irrelevant_responses": state["irrelevant_responses"],
        "reasoning": state["reasoning"]
    }

# Deterministic function to select all relevant tool results based on relevance_threshold
def get_all_relevant_results(state):
    relevant_results = [tool for tool in state["ranking"] if tool not in state["irrelevant_responses"]]
    if relevant_results:
        return [state["tool_responses"][tool] for tool in relevant_results]
    return "None of the tool results are relevant enough."

