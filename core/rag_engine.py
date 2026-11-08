import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_system_prompt(language: str = "english") -> str:
    if language.lower() == "hinglish":
        return """You are an expert meeting assistant. The meeting transcript is in
Hinglish (Hindi/Urdu written in Latin script). Answer the user's question
using ONLY the transcript context below.

STRICT RULES:
- Answer in Hinglish (Hindi/Urdu words in Latin script).
- NEVER use Devanagari script.
- NEVER answer purely in English.
- Keep technical terms (AI, API, deadline, manager) in English.
- If the answer is not in the context, reply exactly:
  "Mujhe meeting transcript mein yeh information nahi mili."

Context from meeting transcript:
{context}"""
    return """You are an expert meeting assistant. Answer the user's question
using ONLY the meeting transcript context below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise. If quoting, mention it.

Context from meeting transcript:
{context}"""


def _make_prompt(language: str):
    return ChatPromptTemplate.from_messages([
        ("system", get_system_prompt(language)),
        ("human", "{question}"),
    ])


def _build_chain_from_store(vector_store, language: str):
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()
    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | _make_prompt(language)
        | llm
        | StrOutputParser()
    )


def build_rag_chain(transcript: str, language: str = "english"):
    return _build_chain_from_store(build_vector_store(transcript), language)


def load_rag_chain(language: str = "english"):
    return _build_chain_from_store(load_vector_store(), language)


def ask_question(rag_chain, question: str) -> str:
    answer = rag_chain.invoke(question)
    print(f"Q: {question}\nA: {answer}")
    return answer