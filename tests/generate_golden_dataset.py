"""
Semi-automatic golden Q&A dataset generator for RAG evaluation.

For each ingested document, samples N chunks and asks Claude to generate
questions that are answerable from that chunk. Outputs a JSONL file that
you then review manually (delete bad lines, edit questions).

Usage:
    python tests/generate_golden_dataset.py [options]

Options:
    --chunks-per-doc  N   How many chunks to sample per document (default: 5)
    --questions-per-chunk N  Questions to generate per chunk (default: 2)
    --output PATH         Output JSONL file (default: tests/golden_dataset.jsonl)
    --model MODEL         Claude model to use (default: claude-haiku-4-5-20251001)
    --review              Interactive review mode: confirm/skip each pair in terminal
    --dry-run             Show what would be sampled without calling Claude

After generation, open tests/golden_dataset.jsonl and:
  - Delete lines with bad/ambiguous questions
  - Edit questions that need clarification
  - Each line is one {question, file_id, filename, chunk_index, source_text} pair
"""

import argparse
import json
import random
import sys
import os

# Allow running from project root or tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import WEAVIATE_URL, HYDE_API_KEY
import weaviate
from urllib.parse import urlparse


# ── Weaviate helpers ────────────────────────────────────────────────────────

def connect_weaviate():
    parsed = urlparse(WEAVIATE_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080
    return weaviate.connect_to_local(host=host, port=port, grpc_port=50051)


def fetch_all_chunks(client) -> list[dict]:
    """Return every chunk stored in Weaviate as a plain dict."""
    collection = client.collections.get("Document")
    results = collection.query.fetch_objects(limit=10_000)
    chunks = []
    for obj in results.objects:
        chunks.append({
            "file_id":     obj.properties["file_id"],
            "filename":    obj.properties.get("filename", ""),
            "chunk_index": obj.properties["chunk_index"],
            "chunk_count": obj.properties.get("chunk_count", 0),
            "page_number": obj.properties.get("page_number", 1),
            "text":        obj.properties["text"],
        })
    return chunks


def group_by_document(chunks: list[dict]) -> dict[str, list[dict]]:
    docs: dict[str, list[dict]] = {}
    for chunk in chunks:
        fid = chunk["file_id"]
        docs.setdefault(fid, []).append(chunk)
    for chunks_list in docs.values():
        chunks_list.sort(key=lambda c: c["chunk_index"])
    return docs


def sample_chunks(doc_chunks: list[dict], n: int) -> list[dict]:
    """
    Sample n chunks spread across the document (beginning, middle, end)
    rather than purely random, to get better coverage.
    """
    if len(doc_chunks) <= n:
        return doc_chunks
    # Ensure we cover start, end and middle evenly
    indices = [int(i * (len(doc_chunks) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
    return [doc_chunks[i] for i in sorted(set(indices))]


# ── Claude question generation ───────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a RAG evaluation expert. Your job is to generate realistic search queries
that a user would type to find the information in the given passage.

Rules:
- Questions must be answerable ONLY from the passage provided — do not ask about
  information not present in the text.
- Write either concise questions or keyword based phrases (2 or 3 words) in a way that a user would ask.
- Vary question types: factual, definitional, how/why, numerical if applicable.
- Return ONLY a JSON array of strings, no explanation.

Example output:
["How does the exploit bypass ASLR?", "webkit type confusion"]
"""

_USER_PROMPT = """\
Generate exactly {n} search queries for the following passage.

Passage (from "{filename}", page {page}):
\"\"\"
{text}
\"\"\"

Return a JSON array of {n} strings.
"""


def generate_questions(chunk: dict, n: int, model: str) -> list[str]:
    import anthropic
    key = HYDE_API_KEY or None
    client = anthropic.Anthropic(api_key=key)

    user_msg = _USER_PROMPT.format(
        n=n,
        filename=chunk["filename"],
        page=chunk["page_number"],
        text=chunk["text"][:2000],  # cap to avoid token overflow
    )

    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        questions = json.loads(raw)
        if isinstance(questions, list):
            return [str(q).strip() for q in questions if str(q).strip()]
    except json.JSONDecodeError:
        # Fallback: split by newline and strip bullets
        lines = [l.lstrip("0123456789.-) ").strip().strip('"') for l in raw.splitlines() if l.strip()]
        return [l for l in lines if l]

    return []


# ── Interactive review ───────────────────────────────────────────────────────

def review_pair(question: str, source_text: str) -> bool:
    """Ask the user whether to keep this Q&A pair. Returns True to keep."""
    print("\n" + "=" * 60)
    print(f"SOURCE TEXT (first 300 chars):\n{source_text[:300]}...")
    print(f"\nQUESTION: {question}")
    while True:
        ans = input("Keep? [y/n/q to quit review mode] ").strip().lower()
        if ans in ("y", ""):
            return True
        if ans == "n":
            return False
        if ans == "q":
            raise KeyboardInterrupt


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate a golden Q&A dataset from ingested documents.")
    parser.add_argument("--chunks-per-doc",       type=int, default=3,    metavar="N")
    parser.add_argument("--questions-per-chunk",  type=int, default=1,    metavar="N")
    parser.add_argument("--output",               default="tests/golden_dataset.jsonl")
    parser.add_argument("--model",                default="claude-haiku-4-5-20251001")
    parser.add_argument("--review",               action="store_true", help="Interactive keep/skip per pair")
    parser.add_argument("--dry-run",              action="store_true", help="Print sampled chunks without calling Claude")
    args = parser.parse_args()

    print(f"Connecting to Weaviate at {WEAVIATE_URL} ...")
    client = connect_weaviate()

    try:
        all_chunks = fetch_all_chunks(client)
    finally:
        client.close()

    if not all_chunks:
        print("No chunks found in Weaviate. Ingest some documents first.")
        sys.exit(1)

    docs = group_by_document(all_chunks)
    print(f"Found {len(docs)} document(s), {len(all_chunks)} total chunks.\n")

    pairs: list[dict] = []

    for file_id, doc_chunks in list(docs.items()):
        filename = doc_chunks[0]["filename"]
        sampled = sample_chunks(doc_chunks, args.chunks_per_doc)
        print(f"  [{filename}] {len(doc_chunks)} chunks → sampling {len(sampled)}")

        if args.dry_run:
            for chunk in sampled:
                print(f"    chunk {chunk['chunk_index']}: {chunk['text'][:80]}...")
            continue

        for chunk in sampled:
            try:
                questions = generate_questions(chunk, args.questions_per_chunk, args.model)
            except Exception as e:
                print(f"    ERROR generating questions for chunk {chunk['chunk_index']}: {e}")
                continue

            for q in questions:
                pair = {
                    "question":    q,
                    "file_id":     file_id,
                    "filename":    filename,
                    "chunk_index": chunk["chunk_index"],
                    "page_number": chunk["page_number"],
                    "source_text": chunk["text"],
                }
                keep = True
                if args.review:
                    try:
                        keep = review_pair(q, chunk["text"])
                    except KeyboardInterrupt:
                        print("\nExiting review mode — remaining pairs will be kept automatically.")
                        args.review = False
                        keep = True

                if keep:
                    pairs.append(pair)
                    print(f"    + chunk {chunk['chunk_index']}: {q}")
                else:
                    print(f"    - skipped: {q}")

    if args.dry_run:
        print("\nDry run complete. No output written.")
        return

    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(pairs)} Q&A pairs to {args.output}")
    print("Next step: open the file and delete/edit any bad questions, then run the evaluator.")


if __name__ == "__main__":
    main()
