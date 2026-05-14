from pathlib import Path

from src.data_acquisition.wildchat import extract_user_prompt_rows, iter_prompt_rows, write_prompt_rows


def test_extract_user_prompt_rows_from_conversation_record() -> None:
    record = {
        "conversation_id": "conv-1",
        "timestamp": "2024-01-01T00:00:00Z",
        "language": "en",
        "model": "gpt-test",
        "conversation": [
            {"role": "user", "content": " Hello there "},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Can you help me study?"},
        ],
    }

    rows = extract_user_prompt_rows(record, 0)

    assert len(rows) == 2
    assert rows[0]["prompt"] == "Hello there"
    assert rows[1]["turn_index"] == 1
    assert rows[1]["message_index"] == 2


def test_iter_prompt_rows_respects_sample_size() -> None:
    records = [
        {
            "conversation_id": "conv-1",
            "conversation": [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ],
        },
        {
            "conversation_id": "conv-2",
            "conversation": [{"role": "user", "content": "three"}],
        },
    ]

    rows = list(iter_prompt_rows(records, sample_size=2))

    assert [row["prompt"] for row in rows] == ["one", "two"]


def test_write_prompt_rows_streams_jsonl() -> None:
    output_dir = Path("tests/_tmp")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "wildchat_sample.jsonl"
    count = write_prompt_rows(({"prompt": str(index)} for index in range(3)), output_file)

    assert count == 3
    assert output_file.read_text(encoding="utf-8").count("\n") == 3
