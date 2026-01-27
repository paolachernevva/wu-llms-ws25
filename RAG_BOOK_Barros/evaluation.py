import re
from langchain_ollama import OllamaLLM
from proyecto import rag_bot  #  RAG pipeline

judge_llm = OllamaLLM(model="llama3.2")

def extract_score(text: str) -> float | None:
    """
    Extract the first number between 0 and 1 (inclusive) from the model output.
    If nothing usable is found, return None.
    """
    m = re.search(r"([01](?:\.\d+)?)", text)
    return float(m.group(1)) if m else None

def score_document_relevance(question: str, contexts: list[str]) -> float | None:
    prompt = f"""
You are grading *document relevance* in a RAG system.
Question:
{question}

Retrieved context chunks:
{"-----".join(contexts)}

On a scale from 0 to 1, where:
- 1 contexts are highly relevant and sufficient,
- 0 means they are irrelevant,
respond with only a single number between 0 and 1.
"""
    resp = judge_llm.invoke(prompt)
    return extract_score(resp)


def score_answer_relevance(question: str, answer: str) -> float | None:
    prompt = f"""
You are grading *answer relevance*.
Question:
{question}

Model answer:
{answer}

On a scale from 0 to 1, where:
- 1 means the answer fully addresses the question,
- 0 means it is off-topic,

respond with only a single number between 0 and 1.
"""
    resp = judge_llm.invoke(prompt)
    return extract_score(resp)


def generate_ground_truth(question: str) -> str:
    """
    Use the LLM to generate a concise reference answer
    based on general world knowledge (not the PDF).
    """
    prompt = f"""
Give a concise, factual answer (3–5 sentences) to the question below,
based on general historical knowledge (do NOT say 'I am an AI'):

Question: {question}
"""
    return judge_llm.invoke(prompt)


def score_answer_correctness(question: str, answer: str, ground_truth: str) -> float | None:
    prompt = f"""
You are grading *answer correctness*.

Question:
{question}

Model answer:
{answer}

Reference (ground-truth) answer:
{ground_truth}

On a scale from 0 to 1, where:
- 1 means the model answer is fully consistent with the reference answer,
- 0 means it contradicts or ignores the reference,

respond with ONLY a single number between 0 and 1.
"""
    resp = judge_llm.invoke(prompt)
    return extract_score(resp)


def score_groundedness(question: str, answer: str, contexts: list[str]) -> float | None:
    prompt = f"""
You are grading *groundedness/faithfulness* of a RAG answer.

Question:
{question}

Retrieved context chunks:
{"\n\n-----\n\n".join(contexts)}

Model answer:
{answer}

On a scale from 0 to 1, where:
- 1 means every important claim in the answer is clearly supported by the context,
- 0 means the answer largely invents information not present in the context,

respond with ONLY a single number between 0 and 1.
"""
    resp = judge_llm.invoke(prompt)
    return extract_score(resp)


def main():
    # 1. Load last question
    with open("last_question.txt", "r") as f:
        question = f.read().strip()

    print(f"\nEvaluating last question:\n{question}\n")

    # 2. Run RAG 
    result = rag_bot(question)
    answer = result["answer"]
    contexts = [d.page_content for d in result["documents"]]

    # 3. Generate ground-truth via LLM
    ground_truth = generate_ground_truth(question)
    print("Generated reference (ground-truth) answer:\n")
    print(ground_truth)
    print("\n" + "=" * 60 + "\n")

    # 4. Compute scores
    doc_rel = score_document_relevance(question, contexts)
    ans_rel = score_answer_relevance(question, answer)
    ans_corr = score_answer_correctness(question, answer, ground_truth)
    grounded = score_groundedness(question, answer, contexts)

    # 5. Show everything nicely
    print("Model answer:\n")
    print(answer)
    print("\nScores (0–1):")
    print(f"  Document relevance   : {doc_rel}")
    print(f"  Answer relevance     : {ans_rel}")
    print(f"  Answer correctness   : {ans_corr}")
    print(f"  Answer groundedness  : {grounded}")


if __name__ == "__main__":
    main()