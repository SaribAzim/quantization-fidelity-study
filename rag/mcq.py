"""
The exam questions, loaded exactly as lm-evaluation-harness sees them.

We evaluate on the same four computer-science MMLU subjects as Phase 2 (412
questions in total, 0-shot), so RAG numbers can be compared directly with the
no-RAG numbers we already have.
"""

from datasets import load_dataset

SUBJECTS = [
    "high_school_computer_science",
    "college_computer_science",
    "computer_security",
    "machine_learning",
]
LETTERS = ["A", "B", "C", "D"]


def load_questions():
    """
    Returns a list of dicts: subject, doc_id, question, choices, answer (0-3).
    doc_id is the question's position inside its subject's test split, the
    same index lm-eval uses, so we can match our answers to its saved ones.
    """
    questions = []
    for subject in SUBJECTS:
        split = load_dataset("cais/mmlu", subject, split="test")
        for doc_id, row in enumerate(split):
            questions.append({
                "subject": subject,
                "doc_id": doc_id,
                "question": row["question"].strip(),
                "choices": list(row["choices"]),
                "answer": int(row["answer"]),
            })
    return questions


def question_block(q):
    """The question and its four options, formatted like lm-eval's MMLU task."""
    lines = [q["question"]]
    lines += [f"{letter}. {choice}" for letter, choice in zip(LETTERS, q["choices"])]
    lines.append("Answer:")
    return "\n".join(lines)


def subject_description(subject):
    """lm-eval's MMLU instruction line for this subject."""
    return f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"


if __name__ == "__main__":
    qs = load_questions()
    print(len(qs), "questions")
    print(subject_description(qs[0]["subject"]) + question_block(qs[0]))
