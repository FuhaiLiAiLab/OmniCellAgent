
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_core.tools import FunctionTool
import re
from autogen_core.models import ModelInfo

# ---- FAKE OMICX FUNCTION FOR TESTING ----
def cell_fetch(cell_type: str = None, disease_type: str = None, tissue_type: str = None):
    if not cell_type or not disease_type or not tissue_type:
        return {"error": "Missing required parameter(s): cell_type, disease_type, or tissue_type."}
    return {
        "cell_type": cell_type,
        "disease_type": disease_type,
        "tissue_type": tissue_type,
        "results": [
            {"gene": "TP53", "score": 8.7, "p_value": 0.0004},
            {"gene": "BRCA1", "score": 7.3, "p_value": 0.0012},
            {"gene": "EGFR", "score": 6.9, "p_value": 0.0021},
            {"gene": "MYC", "score": 6.5, "p_value": 0.0056},
            {"gene": "VEGFA", "score": 6.0, "p_value": 0.0078},
        ]
    }

# ---- ENV & MODEL SETUP ----
load_dotenv(".env")


model_client = OpenAIChatCompletionClient(
    model="gemini-2.5-pro-preview-03-25",
    temperature=0.0,
    # response_format={"type": "json_object"},
    model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="GEMINI_2_5_PRO", structured_output=False)
)

omicx_tool = FunctionTool(
    cell_fetch,
    name="cell_fetch",
    description="Fetches OMICX gene expression data based on specified cell type, disease type, and tissue type. Returns gene names, scores, and p-values."
)

user_input_event = asyncio.Event()
user_response = ""

async def custom_input_func(prompt: str, cancellation_token=None) -> str:
    global user_input_event, user_response
    user_input_event.clear()
    await user_input_event.wait()
    return user_response

omicx_agent = AssistantAgent(
    name="OMICX",
    description="OMICX Data Search Agent",
    model_client=model_client,
    tools=[omicx_tool],
    system_message="You are an OMICX data search agent. Use the OMICX tool to fetch gene expression data based on user-defined cell, disease, and tissue types. If the query lacks sufficient details, clearly state what information is missing and notify the user_proxy agent to obtain clarification."
)

user_proxy_agent = UserProxyAgent("user_proxy", input_func=custom_input_func)

# ---- MESSAGE PARSING ----
def parse_msg(msg):
    match = re.findall(r"(\w+)=('.*?'|\".*?\"|\{.*?\}|None)", msg)
    msg_dict = {k: v.strip("'\"") for k, v in match}
    source = msg_dict.get("source", "other")
    content = msg_dict.get("content", "")
    return source, content

# ---- CHAT FUNCTION ----
async def chat_function(message):
    team = MagenticOneGroupChat(
        [omicx_agent, user_proxy_agent],
        model_client=model_client,
        final_answer_prompt="""
You are a research assistant. Use tools to find the information.
Create a comprehensive report with sections and subsections.
Include factual information and a reference section.
Avoid bullet points.
"""
    )
    async for msg in team.run_stream(task=message):
        source, content = parse_msg(str(msg))
        role = "user" if source in ["user", "user_proxy"] else "assistant"
        if content.strip():
            display = f"{content.strip()} ⌶"
            yield gr.ChatMessage(
                role=role,
                content=display,
                metadata={"title": f"{source}"}
            )

# ---- UI ----
custom_theme = gr.themes.Default(
    primary_hue=gr.themes.colors.indigo,
    font=["Times New Roman", "serif"],
    radius_size="none"
)

with gr.Blocks(theme=custom_theme) as demo:
    gr.Markdown("""
    <h1 style='font-family:Georgia, serif; color:#1c1c3c; text-align:center;'>Multi-Agentic System for Bio-medical Research</h1>
    """)

    chatbot = gr.Chatbot(
        autoscroll=True,
        type="messages",
    )

    msg = gr.Textbox(placeholder="Ask for OMICX gene expression data. Be specific with cell, disease, and tissue.")
    clear = gr.Button("Clear")

    # Floating Scroll Buttons
    gr.HTML("""
    <style>
    #scroll-buttons {
        position: fixed;
        bottom: 20px;
        right: 20px;
        display: flex;
        flex-direction: column;
        gap: 10px;
        z-index: 1000;
    }
    .scroll-btn {
        width: 40px;
        height: 40px;
        border: none;
        border-radius: 50%;
        background-color: #007acc;
        color: white;
        font-size: 20px;
        cursor: pointer;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }
    </style>
    <div id="scroll-buttons">
        <button class="scroll-btn" onclick="window.scrollTo({ top: 0, behavior: 'smooth' });">&#8679;</button>
        <button class="scroll-btn" onclick="window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });">&#8681;</button>
    </div>
    """)

    # ---- Submission Handler ----
    def on_user_submit(message, history):
        global user_response, user_input_event
        user_response = message
        user_input_event.set()
        history.append(gr.ChatMessage(role="user", content=message))
        return "", history

    # ---- Agent Bot ----
    async def bot(history):
        last_msg = history[-1]
        message = last_msg.content if isinstance(last_msg, gr.ChatMessage) else last_msg["content"]
        async for response in chat_function(message):
            history.append(response)
            yield history

    msg.submit(on_user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: [], None, chatbot, queue=False)

demo.launch()
