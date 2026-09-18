import json
import os

from ollama import chat

from parse_data import load_items, get_unclaimed_items, save_result


MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:8b")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ITEMS_FILE = os.path.join(BASE_DIR, "found_items.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "output", "match_result.json")


MATCHING_RULES = """1. Use ONLY the unclaimed items provided in the context.
2. Not all details of an item must match to be a possible match.
   - Item type is the strongest signal.
   - Color, location and date are secondary signals.
3. A match is possible if the item type matches AND at least one
   of (color, location, date) is consistent with the description.
4. If two items are equally plausible, list both.
5. "confidence" reflects how well the description matches:
   - HIGH   : item type + color + location all match.
   - MEDIUM : item type + one secondary attribute match.
   - LOW    : only item type matches, or the description is vague.
6. If no item is a plausible match, return an empty "matches" list."""


def build_prompt(description, available_items):
    system_prompt = (
        "You are a campus lost-and-found assistant.\n"
        "Your task is to identify possible matches between a user's "
        "description of a lost item and the unclaimed items in the database.\n\n"
        "You must respond ONLY with a single valid JSON object. "
        "Do not include markdown code fences, explanations, or extra text.\n\n"
        "The JSON must have exactly this structure:\n"
        "{\n"
        '  "matches": ["ITEM_ID"],\n'
        '  "confidence": "LOW"\n'
        "}\n\n"
        '- "matches" contains all possible matching item IDs (strings).\n'
        '- "confidence" must be exactly one of: LOW, MEDIUM, HIGH.\n'
        '- If there is no match, return an empty "matches" list.'
    )

    context = (
        "=== MATCHING RULES ===\n"
        f"{MATCHING_RULES}\n\n"
        "=== AVAILABLE UNCLAIMED ITEMS (JSON) ===\n"
        f"{json.dumps(available_items, indent=2, ensure_ascii=False)}"
    )

    user_prompt = (
        "User description of the lost item:\n"
        f'"{description}"\n\n'
        "Identify possible matches and return only the JSON result."
    )

    return system_prompt, context, user_prompt


def ask_qwen(system_prompt, context, user_prompt):
    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": context},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={"temperature": 0},
    )
    return response.message.content


def parse_response(response_text):
    text = response_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return json.loads(text)


def validate_result(result, available_items):
    if not isinstance(result, dict):
        return False, "Result must be a dictionary."

    if set(result.keys()) != {"matches", "confidence"}:
        return False, 'Result must contain exactly "matches" and "confidence".'

    matches = result["matches"]
    confidence = result["confidence"]

    if not isinstance(matches, list):
        return False, '"matches" must be a list.'

    if not all(isinstance(item_id, str) for item_id in matches):
        return False, 'All entries in "matches" must be strings.'

    if confidence not in {"LOW", "MEDIUM", "HIGH"}:
        return False, '"confidence" must be LOW, MEDIUM or HIGH.'

    valid_ids = {item["id"] for item in available_items}
    invalid_ids = [i for i in matches if i not in valid_ids]
    if invalid_ids:
        return False, f"Invalid item IDs: {invalid_ids}"

    return True, ""


def display_matches(result, available_items):
    item_by_id = {item["id"]: item for item in available_items}

    print("\nMATCH RESULT")
    print("-" * 50)
    print(f"Confidence: {result['confidence']}")

    if not result["matches"]:
        print("\nNo matches found.")
        return

    print("\nPossible matches:")
    for item_id in result["matches"]:
        item = item_by_id.get(item_id)
        if not item:
            continue
        print(f"\nID: {item['id']}")
        print(f"Item: {item['item']}")
        print(f"Color: {item['color']}")
        print(f"Location: {item['location']}")
        print(f"Date found: {item['date']}")


def main():
    print("CAMPUS LOST-AND-FOUND ASSISTANT")
    print("=" * 50)

    description = input("\nDescribe the item you lost: ").strip()
    if not description:
        print("No description provided.")
        return

    try:
        items = load_items(ITEMS_FILE)
        available_items = get_unclaimed_items(items)
    except Exception as exc:
        print(f"Error loading items: {exc}")
        return

    system_prompt, context, user_prompt = build_prompt(
        description, available_items
    )

    print("\nSearching for possible matches...")

    try:
        response_text = ask_qwen(system_prompt, context, user_prompt)
        result = parse_response(response_text)
    except Exception as exc:
        print(f"Error while asking Qwen or parsing response: {exc}")
        return

    is_valid, error_message = validate_result(result, available_items)
    if not is_valid:
        print(f"Invalid result from Qwen: {error_message}")
        return

    display_matches(result, available_items)

    try:
        save_result(result, OUTPUT_FILE)
        print("\nResult saved to output/match_result.json")
    except Exception as exc:
        print(f"\nError saving result: {exc}")


if __name__ == "__main__":
    main()