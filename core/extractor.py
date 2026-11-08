from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )


def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


_HINGLISH_RULES = (
    "STRICT RULES for output:\n"
    "- Write ONLY in Hinglish (Hindi/Urdu words in Latin script).\n"
    "- NEVER use Devanagari script.\n"
    "- NEVER write purely in English.\n"
    "- Keep technical terms (Deadline, Owner, API, AI) in English.\n"
)


def extract_action_items(transcript: str, language: str = "english") -> str:
    if language.lower() == "hinglish":
        system_prompt = (
            "The transcript is in Hinglish (Latin script). You are an expert "
            "meeting analyst. Extract ALL action items. For each provide:\n"
            "- Task description\n"
            "- Owner (who is responsible)\n"
            "- Deadline (if mentioned, else 'Not specified')\n\n"
            "Format as a numbered list.\n" + _HINGLISH_RULES +
            "\nIf none found, reply exactly: 'Koi action items nahi mile.'"
        )
    else:
        system_prompt = (
            "You are an expert meeting analyst. From the meeting transcript, "
            "extract all action items. For each provide:\n"
            "- Task description\n"
            "- Owner\n"
            "- Deadline (if mentioned, else 'Not specified')\n\n"
            "Format as a numbered list. If none found say 'No action items found.'"
        )
    return build_chain(system_prompt).invoke(transcript)


def extract_key_decisions(transcript: str, language: str = "english") -> str:
    if language.lower() == "hinglish":
        system_prompt = (
            "The transcript is in Hinglish (Latin script). Extract ALL key "
            "decisions made in the meeting. Format as a numbered list.\n"
            + _HINGLISH_RULES +
            "\nIf none found, reply exactly: 'Koi key decisions nahi mile.'"
        )
    else:
        system_prompt = (
            "You are an expert meeting analyst. From the meeting transcript, "
            "extract all key decisions made. Format as a numbered list. "
            "If none found say 'No key decisions found.'"
        )
    return build_chain(system_prompt).invoke(transcript)


def extract_questions(transcript: str, language: str = "english") -> str:
    if language.lower() == "hinglish":
        system_prompt = (
            "The transcript is in Hinglish (Latin script). Extract all "
            "unresolved questions or follow-up topics. Format as a numbered "
            "list.\n" + _HINGLISH_RULES +
            "\nIf none found, reply exactly: 'Koi open questions nahi mile.'"
        )
    else:
        system_prompt = (
            "From the meeting transcript, extract all unresolved questions "
            "or topics needing follow-up. Format as a numbered list. "
            "If none found say 'No open questions found.'"
        )
    return build_chain(system_prompt).invoke(transcript)