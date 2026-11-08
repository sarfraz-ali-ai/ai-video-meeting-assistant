from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    return splitter.split_text(transcript)


# FIX #3: explicit, unambiguous Hinglish contract.
# - Tells the model the input is already Hinglish (post-transliteration).
# - Forbids Devanagari and pure English.
# - Forces the "auto" model to actually follow instructions.
_HINGLISH_MAP = (
    "You are summarising a meeting transcript that is already written in "
    "Hinglish (Hindi/Urdu words in Latin script). Write a concise summary "
    "of this chunk.\n\n"
    "STRICT RULES:\n"
    "- Output MUST be in Hinglish (Latin script only).\n"
    "- NEVER use Devanagari script.\n"
    "- NEVER write purely in English.\n"
    "- Keep technical terms like AI, API, deadline, manager in English.\n"
    "- Do not add preamble like 'Here is the summary'."
)

_HINGLISH_COMBINE = (
    "You are an expert meeting summarizer. Combine the partial Hinglish "
    "summaries below into ONE final professional meeting summary as bullet "
    "points.\n\n"
    "STRICT RULES:\n"
    "- Entire output MUST be Hinglish (Hindi/Urdu in Latin script).\n"
    "- NEVER use Devanagari.\n"
    "- NEVER write purely in English.\n"
    "- Keep technical terms (AI, Agentic AI, deadline, manager, API) in English.\n"
    "- Use bullet points. No preamble.\n\n"
    "Example style:\n"
    "- Meeting mein AI concepts aur Agentic AI ke modules discuss kiye gaye\n"
    "- Deadline 15 tarikh fix ki gayi"
)

_EN_MAP = "Summarize this portion of a meeting transcript concisely in professional English."
_EN_COMBINE = ("You are an expert meeting summarizer. Combine these partial "
               "summaries into one final professional meeting summary in bullet points.")


def summarize(transcript: str, language: str = "english") -> str:
    llm = get_llm()
    hinglish = language.lower() == "hinglish"

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", _HINGLISH_MAP if hinglish else _EN_MAP),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    chunk_summaries = [map_chain.invoke({"text": c}) for c in chunks]
    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        ("system", _HINGLISH_COMBINE if hinglish else _EN_COMBINE),
        ("human", "{text}"),
    ])
    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )
    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    llm = get_llm()
    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system",
             "Based on the meeting transcript, generate a short professional "
             "meeting title (max 8 words). Only return the title, nothing else."),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )
    return title_chain.invoke(transcript[:2000])