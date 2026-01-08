# State dict to manage the flow
state = {
    "query": None,
    "ambiguity_checked": False,
    "clarification_needed": False,
    "clarification": None,
    "final_result": None
}

# Simplified ambiguity check
def check_ambiguity(client, query: str, state, model: str = "gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Determine if the query is ambiguous. Provide clarification questions if needed."},
            {"role": "user", "content": f"Check ambiguity for: '{query}'"}
        ],
        temperature=0.2
    )
    state["ambiguity_checked"] = True
    result = response.choices[0].message.content
    if "clarification" in result.lower() or "ambiguous" in result.lower():
        state["clarification_needed"] = True
    return result

# Simplified final prompt generator
def generate_final_prompt(client, query: str, clarification: str = None, state = {}, model: str = "gpt-4o"):
    prompt = f"Create a clear and actionable prompt based on this query: '{query}'."
    if clarification:
        prompt += f" Clarification provided: '{clarification}'."

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    state["final_result"] = response.choices[0].message.content
    return state["final_result"]

# Example usage (client initialized externally)
# from openai import OpenAI
# client = OpenAI()


if __name__ == "__main__":



    state["query"] = "Build a website"

    ambiguity_result = check_ambiguity(client, state["query"])
    print("Ambiguity check result:\n", ambiguity_result)

    if state["clarification_needed"]:
        # Assume user provides clarification externally
        state["clarification"] = "A personal portfolio site with about and projects sections"

    final_prompt = generate_final_prompt(client, state["query"], state["clarification"])
    print("Final prompt:\n", final_prompt)

    # Print the final state dict
    print("\nFinal state:", state)