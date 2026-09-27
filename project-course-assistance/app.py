import os
from dataclasses import asdict
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from assistant import create_assistant
from db_save import save_conversation
from db_feedback import save_feedback
from judge import evaluate_relevance
from db_query import get_conversations, get_stats

load_dotenv()

st.set_page_config(page_title="Course Assistant", page_icon="🤖", layout="wide")

COURSES = ["llm", "mlops", "machine-learning", "data-engineering"]
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-max-2026-06-08"

DEFAULT_INPUT_PRICE = 0.96
DEFAULT_OUTPUT_PRICE = 3.84

MODEL_PRESETS = {
    "qwen3.7-max-2026-06-08": {"input": 0.96, "output": 3.84},
    "qwen-plus": {"input": 0.40, "output": 1.20},
    "qwen-turbo": {"input": 0.05, "output": 0.20},
    "qwen-max": {"input": 1.60, "output": 6.40},
}

# Initialize session state for API credentials, model, and token pricing
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")
if "model_name" not in st.session_state:
    st.session_state.model_name = os.getenv("LLM_MODEL", DEFAULT_MODEL)
if "base_url" not in st.session_state:
    st.session_state.base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
if "input_token_price" not in st.session_state:
    st.session_state.input_token_price = float(os.getenv("INPUT_TOKEN_PRICE", DEFAULT_INPUT_PRICE))
if "output_token_price" not in st.session_state:
    st.session_state.output_token_price = float(os.getenv("OUTPUT_TOKEN_PRICE", DEFAULT_OUTPUT_PRICE))


@st.cache_resource
def get_assistant(api_key, model, base_url, input_price, output_price):
    return create_assistant(
        api_key=api_key,
        model=model,
        base_url=base_url,
        input_price_per_million=input_price,
        output_price_per_million=output_price
    )


def chat_page():
    header_col1, header_col2, header_col3 = st.columns([3, 1, 1])
    with header_col1:
        st.title("Course Assistant")
        st.caption(
            f"🤖 **Model:** `{st.session_state.model_name}` | 💰 **Pricing:** `${st.session_state.input_token_price:.2f}` in / `${st.session_state.output_token_price:.2f}` out (per 1M) | 🌐 **API Base:** `{st.session_state.base_url}`"
        )
    with header_col2:
        st.write("")
        if st.button("📊 View Dashboard", use_container_width=True):
            st.switch_page(page_dashboard)
    with header_col3:
        st.write("")
        if st.button("⚙️ Settings", use_container_width=True):
            st.switch_page(page_settings)

    if not st.session_state.api_key:
        st.warning("⚠️ No API Key configured. Please go to Settings to enter your API Key before asking questions.")
        if st.button("👉 Go to Settings"):
            st.switch_page(page_settings)
        return

    course = st.selectbox("Select Course:", COURSES, index=0)

    user_input = st.text_input(
        "Enter your question:",
        placeholder="e.g. How do I join the course?",
        key="user_question_input"
    )

    ask_clicked = st.button("🚀 Ask Question", type="primary", use_container_width=True)

    if ask_clicked:
        if not user_input.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Thinking..."):
            assistant = get_assistant(
                st.session_state.api_key,
                st.session_state.model_name,
                st.session_state.base_url,
                st.session_state.input_token_price,
                st.session_state.output_token_price
            )
            assistant.course = course
            assistant.model = st.session_state.model_name
            assistant.input_price_per_million = st.session_state.input_token_price
            assistant.output_price_per_million = st.session_state.output_token_price

            answer = assistant.rag(user_input)
            record = assistant.last_call

            conversation_id = save_conversation(record, user_input, course)
            st.session_state.conversation_id = conversation_id

            relevance, explanation = evaluate_relevance(
                user_input,
                answer,
                model=st.session_state.model_name,
                api_key=st.session_state.api_key,
                base_url=st.session_state.base_url
            )
            save_feedback(conversation_id, "judge",
                          relevance=relevance, explanation=explanation)

            st.session_state.last_qa = {
                "question": user_input,
                "answer": answer,
                "record": record,
                "relevance": relevance,
                "explanation": explanation,
                "conversation_id": conversation_id,
            }
            st.session_state.user_feedback_submitted = None

    if st.session_state.get("last_qa"):
        last_qa = st.session_state.last_qa

        st.markdown("---")
        with st.chat_message("user"):
            st.markdown(f"**{last_qa['question']}**")

        with st.chat_message("assistant"):
            st.markdown(last_qa["answer"])

            rec = last_qa["record"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("⏱️ Response Time", f"{rec.response_time:.2f}s")
            m2.metric("📥 Prompt Tokens", rec.prompt_tokens)
            m3.metric("📤 Completion Tokens", rec.completion_tokens)
            m4.metric("💰 Cost", f"${rec.cost:.4f}")

            relevance = last_qa["relevance"]
            rel_badge = {
                "RELEVANT": "🟢 **RELEVANT**",
                "PARTLY_RELEVANT": "🟡 **PARTLY RELEVANT**",
                "NON_RELEVANT": "🔴 **NON RELEVANT**"
            }.get(relevance, f"**{relevance}**")

            with st.expander(f"⚖️ LLM Judge Evaluation: {rel_badge}", expanded=True):
                st.write(f"**Explanation:** {last_qa['explanation']}")

            st.markdown("##### Was this answer helpful?")
            rate_col1, rate_col2 = st.columns(2)
            with rate_col1:
                if st.button("👍 Helpful (+1)", use_container_width=True, key="btn_thumbs_up"):
                    cid = last_qa["conversation_id"]
                    save_feedback(cid, "user", score=1)
                    st.session_state.user_feedback_submitted = 1

            with rate_col2:
                if st.button("👎 Not Helpful (-1)", use_container_width=True, key="btn_thumbs_down"):
                    cid = last_qa["conversation_id"]
                    save_feedback(cid, "user", score=-1)
                    st.session_state.user_feedback_submitted = -1

            if st.session_state.get("user_feedback_submitted") == 1:
                st.success("✅ Thank you for your positive feedback (+1)!")
            elif st.session_state.get("user_feedback_submitted") == -1:
                st.info("🙏 Thank you for your feedback (-1), we will improve our answers.")



def dashboard_page():
    header_col1, header_col2, header_col3 = st.columns([3, 1, 1])
    with header_col1:
        st.title("Course Assistant Dashboard")
    with header_col2:
        st.write("")
        if st.button("💬 Back to Chat", use_container_width=True):
            st.switch_page(page_chat)
    with header_col3:
        st.write("")
        if st.button("⚙️ Settings", use_container_width=True):
            st.switch_page(page_settings)

    stats = get_stats()

    col1, col2, col3, col4 = st.columns(4)
    avg_time = stats.avg_response_time or 0.0
    total_cost = stats.total_cost or 0.0
    avg_tokens = stats.avg_tokens or 0

    col1.metric("Total conversations", stats.total or 0)
    col2.metric("Avg response time", f"{avg_time:.2f}s")
    col3.metric("Total cost", f"${total_cost:.4f}")
    col4.metric("Avg tokens", f"{avg_tokens:.0f}")


    records = get_conversations(limit=100)
    df = pd.DataFrame([asdict(r) for r in records])

    st.subheader("Cost over time")
    if not df.empty and "timestamp" in df.columns and "cost" in df.columns:
        st.line_chart(df, x="timestamp", y="cost")
    else:
        st.info("No cost data available yet.")

    st.subheader("Response time over time")
    if not df.empty and "timestamp" in df.columns and "response_time" in df.columns:
        st.line_chart(df, x="timestamp", y="response_time")
    else:
        st.info("No response time data available yet.")

    st.subheader("Recent conversations")
    recent_records = get_conversations(limit=20)

    for record in recent_records:
        st.write(f"**{record.prompt[:80]}...**")
        st.write(f"{record.answer[:200]}...")
        st.write(f"Time: {record.response_time:.2f}s | Cost: ${record.cost:.4f}")
        st.divider()


def settings_page():
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.title("⚙️ LLM & API Settings")
    with header_col2:
        st.write("")
        if st.button("💬 Back to Chat", use_container_width=True):
            st.switch_page(page_chat)

    st.markdown(
        "Configure your LLM model and API credentials here. These settings are dynamically applied to the Course Assistant and Relevance Judge without restarting the application."
    )

    with st.form("settings_form"):
        api_key_input = st.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
            help="Your API key (e.g. Alibaba Cloud DashScope API key)"
        )

        model_input = st.text_input(
            "Model Name",
            value=st.session_state.model_name,
            help="e.g. qwen3.7-max-2026-06-08, qwen-plus, qwen-turbo, qwen-max"
        )

        base_url_input = st.text_input(
            "Base URL",
            value=st.session_state.base_url,
            help="Default: https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )

        st.markdown("##### 🪙 Token Pricing ($ / 1 Million Tokens)")
        price_col1, price_col2 = st.columns(2)
        with price_col1:
            input_price_input = st.number_input(
                "Input Token Price ($)",
                min_value=0.0,
                value=float(st.session_state.input_token_price),
                step=0.01,
                format="%.4f",
                help="Price per 1M input prompt tokens"
            )
        with price_col2:
            output_price_input = st.number_input(
                "Output Token Price ($)",
                min_value=0.0,
                value=float(st.session_state.output_token_price),
                step=0.01,
                format="%.4f",
                help="Price per 1M completion output tokens"
            )

        submitted = st.form_submit_button("💾 Save Settings", use_container_width=True)
        if submitted:
            st.session_state.api_key = api_key_input.strip()
            st.session_state.model_name = model_input.strip()
            st.session_state.base_url = base_url_input.strip()
            st.session_state.input_token_price = float(input_price_input)
            st.session_state.output_token_price = float(output_price_input)
            st.cache_resource.clear()
            st.success("✅ Settings updated successfully!")

    st.subheader("Quick Model Presets")
    st.caption("Click to load default model name and pricing presets:")
    preset_cols = st.columns(4)
    for i, (m_name, p_info) in enumerate(MODEL_PRESETS.items()):
        with preset_cols[i]:
            btn_label = f"**{m_name.split('-202')[0]}**\n\n`${p_info['input']:.2f}` / `${p_info['output']:.2f}`"
            if st.button(btn_label, use_container_width=True, key=f"preset_{i}"):
                st.session_state.model_name = m_name
                st.session_state.input_token_price = float(p_info["input"])
                st.session_state.output_token_price = float(p_info["output"])
                st.cache_resource.clear()
                st.rerun()


page_chat = st.Page(chat_page, title="Course Assistant", icon="💬", default=True)
page_dashboard = st.Page(dashboard_page, title="Dashboard", icon="📊", url_path="dashboard")
page_settings = st.Page(settings_page, title="Settings", icon="⚙️", url_path="settings")

pg = st.navigation([page_chat, page_dashboard, page_settings])
pg.run()