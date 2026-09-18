
#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Uses the vision-capable DeepSeek Flash model named
    `deepseek-v4-flash-vision-exp`.
    """
    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0.0,
    )

    system_prompt = (
        "You are an expert financial assistant processing supermarket receipts.\n"
        "Read the receipt image carefully and extract the following fields.\n"
        "- final_total: the final payment amount AFTER the ROUNDING adjustment line.\n"
        "- subtotal: the SUBTOTAL amount BEFORE the ROUNDING adjustment. "
        "If there is no SUBTOTAL line, use the net amount after all discounts but before rounding.\n"
        "- discount_total: the sum of the absolute values of ALL discount/promotion/coupon/member/app/percentage-off lines. "
        "This must be a positive number. Do NOT include the ROUNDING adjustment.\n"
        "- original_amount: subtotal + discount_total. Do NOT add back ROUNDING.\n\n"
        "Return ONLY one valid JSON object, without markdown or extra text:\n"
        '{{"final_total": 102.30, "subtotal": 102.31, "discount_total": 5.39, "original_amount": 107.70}}'
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "Extract the receipt amounts. Return only JSON.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_url}"},
                    },
                ],
            ),
        ]
    )

    return prompt | llm


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string."""
    if not images:
        return {QUERY_1: "HK$0.00", QUERY_2: "HK$0.00"}

    def to_decimal(value: Any) -> Decimal:
        if value is None:
            return Decimal("0.00")
        text = str(value).replace(",", "").replace("HK$", "").replace("$", "").strip()
        return Decimal(text)

    total_spent = Decimal("0.00")
    total_original = Decimal("0.00")

    for img_path in images:
        result = chain.invoke({"image_url": image_data_url(img_path)})
        text = response_text(result)

        try:
            cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
            data = json.loads(cleaned)

            spent = to_decimal(data.get("final_total", data.get("amount_spent")))
            if "subtotal" in data and "discount_total" in data:
                original = to_decimal(data["subtotal"]) + to_decimal(data["discount_total"])
            else:
                original = to_decimal(data.get("original_amount"))

            total_spent += spent
            total_original += original
        except (json.JSONDecodeError, InvalidOperation, TypeError, ValueError):
            matches = _MONEY_RE.findall(text)
            if matches:
                try:
                    total_spent += to_decimal(matches[0])
                except InvalidOperation:
                    pass
                if len(matches) >= 2:
                    try:
                        total_original += to_decimal(matches[1])
                    except InvalidOperation:
                        pass

    return {
        QUERY_1: f"HK${total_spent.quantize(Decimal('0.01')):.2f}",
        QUERY_2: f"HK${total_original.quantize(Decimal('0.01')):.2f}",
    }
_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)

def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
