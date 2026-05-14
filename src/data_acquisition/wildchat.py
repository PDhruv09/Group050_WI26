"""Acquire WildChat records from Hugging Face and save prompt-level raw data."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing.io import write_dataset
from src.preprocessing.metadata import normalize_whitespace


DEFAULT_DATASET_NAME = "allenai/WildChat"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT = "data/raw/wildchat_prompts_raw.jsonl"


def make_json_safe(value: Any) -> Any:
    """Convert common dataset scalar values into JSON-serializable values."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    return value


def load_huggingface_dataset(dataset_name: str, split: str, streaming: bool) -> Iterable[dict[str, Any]]:
    """Load a Hugging Face dataset lazily so the dependency is only needed for acquisition."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "The 'datasets' package is required for WildChat acquisition. "
            "Install it with: pip install -r requirements.txt"
        ) from error

    return load_dataset(dataset_name, split=split, streaming=streaming)


def coerce_conversation(conversation: Any) -> list[dict[str, Any]]:
    """Coerce common WildChat conversation shapes into a list of message dictionaries."""
    if conversation is None:
        return []

    if hasattr(conversation, "tolist"):
        conversation = conversation.tolist()

    if isinstance(conversation, dict) and "messages" in conversation:
        conversation = conversation["messages"]

    if not isinstance(conversation, list):
        return []

    return [message for message in conversation if isinstance(message, dict)]


def extract_user_prompt_rows(
    record: dict[str, Any],
    fallback_index: int,
    source_dataset: str = DEFAULT_DATASET_NAME,
) -> list[dict[str, Any]]:
    """Extract one prompt-level row for each user message in a conversation record."""
    conversation = coerce_conversation(record.get("conversation"))
    conversation_id = (
        record.get("conversation_id")
        or record.get("conversation_hash")
        or record.get("id")
        or f"wildchat_{fallback_index}"
    )
    timestamp = record.get("timestamp") or record.get("created_at") or record.get("date")
    language = record.get("language")
    model = record.get("model")

    rows = []
    user_turn = 0
    for message_index, message in enumerate(conversation):
        role = message.get("role") or message.get("from")
        if role != "user":
            continue

        prompt = normalize_whitespace(message.get("content") or message.get("value") or "")
        if not prompt:
            continue

        rows.append(
            {
                "conversation_id": conversation_id,
                "turn_index": user_turn,
                "message_index": message_index,
                "prompt": prompt,
                "timestamp": timestamp,
                "language": language,
                "model": model,
                "source_dataset": source_dataset,
            }
        )
        user_turn += 1

    return rows


def iter_prompt_rows(
    records: Iterable[dict[str, Any]],
    source_dataset: str = DEFAULT_DATASET_NAME,
    max_conversations: int | None = None,
    sample_size: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield prompt-level rows from conversation-level records."""
    emitted = 0
    for record_index, record in enumerate(records):
        if max_conversations is not None and record_index >= max_conversations:
            break

        for row in extract_user_prompt_rows(record, record_index, source_dataset):
            yield row
            emitted += 1
            if sample_size is not None and emitted >= sample_size:
                return


def write_prompt_rows(rows: Iterable[dict[str, Any]], output_file: Path) -> int:
    """Write prompt rows, streaming JSONL/NDJSON to avoid high memory usage."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_file.suffix.lower()

    if suffix in {".jsonl", ".ndjson"}:
        count = 0
        with output_file.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(make_json_safe(row), ensure_ascii=True))
                file.write("\n")
                count += 1
        return count

    frame = pd.DataFrame(rows)
    write_dataset(frame, output_file)
    return int(len(frame))


def acquire_wildchat(
    output_file: Path,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = DEFAULT_SPLIT,
    streaming: bool = False,
    max_conversations: int | None = None,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Download/extract WildChat prompts and write them to a local raw file."""
    if sample_size == 0:
        sample_size = None

    dataset = load_huggingface_dataset(dataset_name, split, streaming)
    rows = iter_prompt_rows(
        dataset,
        source_dataset=dataset_name,
        max_conversations=max_conversations,
        sample_size=sample_size,
    )
    prompt_rows = write_prompt_rows(rows, output_file)

    return {
        "dataset_name": dataset_name,
        "split": split,
        "streaming": streaming,
        "output_file": str(output_file),
        "prompt_rows": prompt_rows,
        "max_conversations": max_conversations,
        "sample_size": sample_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire WildChat prompt-level raw data.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Maximum prompt rows to save. Use 0 for no prompt limit.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream records from Hugging Face instead of downloading the split eagerly.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Deprecated compatibility flag. Non-streaming is now the default.",
    )
    args = parser.parse_args()

    summary = acquire_wildchat(
        output_file=Path(args.output),
        dataset_name=args.dataset_name,
        split=args.split,
        streaming=args.streaming and not args.no_streaming,
        max_conversations=args.max_conversations,
        sample_size=args.sample_size,
    )

    print("WildChat acquisition complete.")
    print(f"Dataset: {summary['dataset_name']}")
    print(f"Split: {summary['split']}")
    print(f"Prompt rows: {summary['prompt_rows']}")
    print(f"Output: {summary['output_file']}")


if __name__ == "__main__":
    main()
