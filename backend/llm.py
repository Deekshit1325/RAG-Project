import os
from groq import Groq
from dotenv import load_dotenv


def build_prompt(query, context_chunks):
    """
    Builds the prompt that we send to the LLM.
    Context chunks are stuffed in so the model can ground its answer.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        src = chunk["source"]
        page = chunk["page"]
        context_parts.append(f"[Source: {src}, Page/Section: {page}]\n{chunk['text']}")

    context_str = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant that answers questions based on the provided documents.
Use ONLY the information from the context below to answer. If the answer isn't in the context, say so.
Always cite which document and page/section your answer came from.

CONTEXT:
{context_str}

QUESTION: {query}

Provide a clear, detailed answer with citations in the format [Source: filename, Page: X]."""

    return prompt


def generate_answer(query, context_chunks):
    """
    Generates an answer using Groq (Llama model).
    Returns the generated answer string.
    """
    load_dotenv(override=True)
    prompt = build_prompt(query, context_chunks)

    try:
        return _call_groq(prompt)
    except Exception as e:
        error_msg = str(e).lower()
        print(f"[LLM Error] {e}")
        if "rate" in error_msg or "429" in error_msg:
            return "⚠️ Rate limit hit. Wait a moment and try again."
        return f"⚠️ LLM API error: {e}"


def _call_groq(prompt):
    api_key = os.getenv("GROQ_API_KEY", "")

    if not api_key or api_key == "your-groq-key-here":
        return "⚠️ Groq API key not set. Add it to your .env file."

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2048,
    )
    return response.choices[0].message.content
