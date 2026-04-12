
import re
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Attribute keys tracked in Auto-ReID (matching paper's structural prompt)
# ---------------------------------------------------------------------------
ATTR_KEYS = [
    "Gender",
    "Age",
    "Hair",
    "Upper",       # upper clothing type/color/pattern
    "Lower",       # lower clothing type/color
    "Footwear",
    "Bag",         # carried items / accessories
]

# ---------------------------------------------------------------------------
# Lightweight rule-based parser
# Used as fallback when the VLM is not available for parsing.
# ---------------------------------------------------------------------------

# Gender cues
_GENDER_FEMALE = re.compile(
    r'\b(woman|female|girl|lady|she|her)\b', re.IGNORECASE)
_GENDER_MALE = re.compile(
    r'\b(man|male|boy|guy|he|his)\b', re.IGNORECASE)

# Age cues
_AGE_PATTERN = re.compile(
    r'\b(teen|teenager|young|mid[- ]?20s?|late[- ]?20s?|early[- ]?30s?|'
    r'mid[- ]?30s?|late[- ]?30s?|40s?|50s?|old|elderly)\b',
    re.IGNORECASE)

# Hair cues
_HAIR_PATTERN = re.compile(
    r'\b(?:(?:long|short|medium|shoulder[- ]length|bald)\s+)?'
    r'(?:(?:dark|black|brown|blonde|gray|white|red)\s+)?hair\b',
    re.IGNORECASE)

# Bag / carried items
_BAG_POSITIVE = re.compile(
    r'\b(backpack|handbag|shoulder[- ]bag|luggage|suitcase|bag|purse|'
    r'carrying|carries)\b', re.IGNORECASE)
_BAG_NEGATIVE = re.compile(
    r'\b(no[- ]bag|without[- ]bag|no[- ]backpack|empty[- ]hand)\b',
    re.IGNORECASE)

# Upper clothing colours / types (heuristic)
_UPPER_PATTERN = re.compile(
    r'\b(?:wears?|wearing|dressed in)\s+(?:a\s+)?'
    r'([\w\s\-,]+?)(?:\s+(?:shirt|jacket|coat|top|hoodie|sweater|blouse|'
    r'tshirt|t-shirt|dress|vest))',
    re.IGNORECASE)

# Lower clothing
_LOWER_PATTERN = re.compile(
    r'\b(?:(?:black|white|gray|blue|red|green|yellow|brown|dark|light)\s+)?'
    r'(pants?|jeans?|shorts?|skirt|trousers?|leggings?)\b',
    re.IGNORECASE)

# Footwear
_FOOTWEAR_PATTERN = re.compile(
    r'\b(?:(?:black|white|gray|blue|red|green|yellow|brown|dark|light)\s+)?'
    r'(shoes?|sneakers?|boots?|sandals?|heels?|loafers?|flats?)\b',
    re.IGNORECASE)


def parse_attributes_rule_based(text: str) -> Dict[str, str]:
    """
    Rule-based attribute parser used as lightweight fallback.
    Returns a dict mapping attribute keys to extracted values (or 'unknown').
    """
    attrs: Dict[str, str] = {k: "unknown" for k in ATTR_KEYS}

    # Gender
    if _GENDER_FEMALE.search(text):
        attrs["Gender"] = "Female"
    elif _GENDER_MALE.search(text):
        attrs["Gender"] = "Male"

    # Age
    m = _AGE_PATTERN.search(text)
    if m:
        attrs["Age"] = m.group(0)

    # Hair
    m = _HAIR_PATTERN.search(text)
    if m:
        attrs["Hair"] = m.group(0)

    # Upper clothing
    m = _UPPER_PATTERN.search(text)
    if m:
        attrs["Upper"] = m.group(1).strip()
    else:
        # Fallback: look for colour + garment
        colour_garment = re.search(
            r'\b((?:black|white|gray|blue|red|green|yellow|brown|dark|light)\s+'
            r'(?:shirt|jacket|coat|top|hoodie|sweater|blouse|t-shirt|dress|vest))\b',
            text, re.IGNORECASE)
        if colour_garment:
            attrs["Upper"] = colour_garment.group(1)

    # Lower clothing
    m = _LOWER_PATTERN.search(text)
    if m:
        attrs["Lower"] = m.group(0)

    # Footwear
    m = _FOOTWEAR_PATTERN.search(text)
    if m:
        attrs["Footwear"] = m.group(0)

    # Bag
    if _BAG_NEGATIVE.search(text):
        attrs["Bag"] = "None"
    elif _BAG_POSITIVE.search(text):
        m = _BAG_POSITIVE.search(text)
        attrs["Bag"] = m.group(1) if m else "bag"

    return attrs


def parse_attributes_from_vlm_response(vlm_output: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for line in vlm_output.strip().splitlines():
        if ':' in line:
            key, _, val = line.partition(':')
            key = key.strip().title()
            val = val.strip()
            if key in ATTR_KEYS and val:
                attrs[key] = val

    # Fill missing keys
    for k in ATTR_KEYS:
        if k not in attrs:
            attrs[k] = "unknown"

    # If barely anything was parsed, fall back to rule-based
    known = sum(1 for v in attrs.values() if v != "unknown")
    if known < 2:
        attrs = parse_attributes_rule_based(vlm_output)

    return attrs


def build_attribute_question(key: str, value: str) -> str:
    templates = {
        "Gender":   "Is the person in the image {val}?",
        "Age":      "Does the person appear to be in their {val}?",
        "Hair":     "Does the person have {val} hair?",
        "Upper":    "Is the person wearing {val}?",
        "Lower":    "Is the person wearing {val}?",
        "Footwear": "Is the person wearing {val}?",
        "Bag":      "Is the person {val}?",
    }
    template = templates.get(key, "Does the person have {val}?")
    if key == "Bag":
        if value.lower() == "none":
            q = "Is the person NOT carrying any bag or backpack?"
        else:
            q = template.format(val=f"carrying a {value}")
    else:
        q = template.format(val=value)
    return q


def build_negative_constraint(key: str, value: str) -> str:
    if key == "Bag" and value.lower() != "none":
        return f"Exclude candidates carrying a {value}."
    elif key == "Bag" and value.lower() == "none":
        return "Exclude candidates carrying bags."
    return f"Exclude candidates with {value} {key.lower()}."


def build_emphasis_constraint(key: str, value: str) -> str:
    if key == "Bag" and value.lower() == "none":
        return "Prioritize candidates who are NOT carrying any bag."
    elif key == "Bag":
        return f"Prioritize candidates carrying a {value}."
    return f"Prioritize candidates wearing {value}." if key in (
        "Upper", "Lower", "Footwear"
    ) else f"Prioritize candidates with {value} {key.lower()}."
