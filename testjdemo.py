import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import os
import ast
import json
import base64
import requests
import tempfile
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="DOC Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# SCHEMA DEFINITION
# =====================================================
SCHEMA = {
    "First_Name": "None",
    "Last_Name": "None",
    "Salutation": "None",
    "Gender": "None",
    "Date_Of_Birth": "None",
    "Birth_City": "None",
    "Birth_Province_or_State": "None",
    "Birth_Country": "None",
    "Tax_Code": "None",
    "Citizenship_Country": "None",
    "Marital_Status": "None",
    "ID_Info": {
        "Passport_Number": "None",
        "ID_Card_Number": "None",
        "Document_Issue_Date": "None",
        "Document_Expiry_Date": "None"
    },
    "Street_Address": "None",
    "City": "None",
    "State_or_Province": "None",
    "Zip_or_Postal_Code": "None",
    "Country": "None",
    "Phone": "None",
    "Mobile": "None",
    "Fax": "None",
    "Primary_Email": "None",
    "Education_History": [
        {
            "School_Name": "None",
            "Degree": "None"
        }
    ],
    "Primary_Language": "None",
    "Languages": [

    ],
    "Education_Level": "None",
    "Diocese": "None",
    "Bishop_Email": "None",
    "Bishop_Name": "None",
    "Seminary_Name": "None",
    "Seminary_Address": "None",
    "Seminary_Email": "None",
    "Religious_Status": "None",
    "Religious_Order": "None",
    "Ordination_Date": "None",
    "Document_Metadata": {
        "Document_Type": "None",
        "Document_Date": "None",
        "Document_Year": "None"
    }
}

# =====================================================
# ZOHO CATALYST & LM STUDIO CONFIGURATION
# =====================================================
VLM_URL = "https://api.catalyst.zoho.com/quickml/v1/project/35939000000182003/vlm/chat"
TOKEN_URL = "https://accounts.zoho.com/oauth/v2/token"
LLM_URL = "https://api.catalyst.zoho.com/quickml/v2/project/35939000000182003/llm/chat"

CATALYST_ORG = "914134238"
MODEL_NAME = "VL-Qwen2.5-7B"
CONSOLIDATION_MODEL_NAME = "crm-di-qwen_text_14b-fp8-it"

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")

OCR_MAX_WORKERS = 8
CONSOLIDATION_MAX_WORKERS = 6
CONSOLIDATION_BATCH_SIZE = 5

# Toggle to show/hide the "Total Extraction Time" indicator in the UI.
SHOW_EXTRACTION_TIME = False

# Local folder (inside the project) where uploaded PDFs are persisted.
LOCAL_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploaded_files")
os.makedirs(LOCAL_UPLOAD_DIR, exist_ok=True)

# =====================================================
# COMPREHENSIVE EXTRACTION PROMPTS
# =====================================================
OCR_SYSTEM_PROMPT = """
You are a strict, automated data conversion extraction utility. Your response MUST begin with '{' and end with '}' and contain NO other text, conversational filler, or markdown code fences.

CORE OUTPUT RULES:
- Return VALID JSON ONLY.
- Follow the schema EXACTLY.
- Never add, rename, remove, reorder, or translate schema keys.
- Every schema field MUST exist.
- Missing or unclear values -> null.
- Arrays must always be arrays []. Never null.
- Never guess, infer, or hallucinate values.

HARD NULL ENFORCEMENT:
- If a field is not explicitly and clearly supported by nearby text, return null.
- Do NOT repair partial OCR.
- Do NOT complete fragmented words.
- Uncertain extraction MUST be null.
- Do NOT infer countries, states, genders, degrees, or education levels.
- Do NOT convert contextual clues into structured values.
- If any field does not have an explicit, explicitly labeled target in the text, it MUST be null. Never backfill empty fields using text from an unrelated section.

ANTI-HALLUCINATION RULES:
- Never create synthetic dates like '1997-01-01'.
- Never transform partial dates into ISO format.
- Strictly Never transform nationality adjectives into countries (example: 'AMERICAN' != 'USA').
- Never promote academic labels into Education_Level.
- Never expand religious movement names.

EDUCATION VERBATIM ENFORCEMENT:
- School_Name must be copied EXACTLY as present in OCR output.
- Do NOT correct spelling, capitalization, punctuation, or accents.
- Do NOT normalize institution names.
- Example: 'UNIVERSITÀ' PONTiFlCIA REGINA APOSTOLORUM' must remain unchanged.

FIELD LOCK RULES:
- Address fields MUST contain only address information.
- Birth fields MUST contain only birth information.
- Education fields MUST contain only education information.
- Religious fields MUST contain only explicit religious information.
- Never copy one value into multiple unrelated fields.

FIELD MAPPING & NORMALIZATION:
- 'Cognome' -> Last_Name.
- 'Nome' -> First_Name.
- 'Stato civile' -> Marital_Status.
- If marital value is 'Consagrada' or 'Consacrata', normalize exactly to 'Single (Consagrada)'.
- Normalize all HOME SCHOOLLED variants to exactly 'Home Schooled'.
- Preserve original email spelling and date formatting.
- Do NOT normalize date formats.

## GENDER NORMALIZATION
- `Gender`: You MUST normalize the value to exactly one of these three options: **Male**, **Female**, or **Others**.
- If the text shows variants like "M", "Maschio", or "Masculino", change it to **Male**.
- If the text shows variants like "F", "Femmina", or "Femenino", change it to **Female**.
- If it cannot be determined or is missing, set it to `null`. Do not guess.

## SALUTATION RULES
- `Salutation` MUST be an actual honorific/title (e.g., "Mr", "Mrs", "Ms", "Dr", "Rev", "Fr", "Sr", "Sra", "Don", "Doña", "S.E.R. Mons.").
- Do NOT misidentify prepositions, conjunctions, or connecting words (such as "Cum", "Et", "De", "Di", "Da", "Del", "Della", "Y", "And", "Con") as a salutation. If only such a word appears where a salutation would be expected, set `Salutation` to `null`.
- If no explicit honorific/title is present in the text, set `Salutation` to `null`. Never guess.

## LANGUAGE RULES
- `Primary_Language`: Detect the primary language the document itself is written in (e.g., "English", "Italian", "Spanish"). 
- `Languages`: Put that same detected language into this array as a single string (e.g., `["English"]`).
- **Rule:** If the document has no readable text (only signatures or stamps), set `Primary_Language` to `null` and `Languages` to `[]`. Do not guess.

GEOGRAPHY RULES:
Strictly never mention birth Country in place of street address.
- (Strictly) If a birth city value is actually a country name (USA, India, France, Italy), move it to Birth_Country and set Birth_City=null.
- States/provinces are NOT countries.
- (Strictly) Never map values like 'Rhode Island', 'WV', 'OH', or 'Oregon' into Birth_Country.
- (Strictly) Do not reuse address or school locations as birth data.
- (Strictly) US states are NEVER countries.
- (Strictly) (Examples: Oregon, Rhode Island, WV, OH are NOT Birth_Country.
- (Strictly) Birth_City, Birth_Province_or_State, and Birth_Country MUST ONLY be extracted if the source text explicitly labels them as birth information (e.g., "Place of Birth", "Born at", "Nato a").
- NEVER use a school location, current address, or document signing location to populate Birth_City, Birth_Province_or_State, or Birth_Country. If an explicit birth declaration is missing, these fields MUST be null.

GEOGRAPHY FIELD LOCK:## RESIDENTIAL ADDRESS RULES
Extract address fields ONLY if explicitly labeled as a current, residential, or home address. Otherwise, leave them null.

Strictly split the address into these components:
- `Street_Address`: Extract ONLY the house number, street name, apartment/suite number, and road name (e.g., "1601 Main Street"). Locally strip and REMOVE the city, state, ZIP code, and country from this specific field.
- `City`: The city name only (e.g., "Wellsburg").
- `State_or_Province`: The state or province name/abbreviation only (e.g., "WV").
- `Zip_or_Postal_Code`: The postal/ZIP code number only (e.g., "26070").
- `Country`: The country name only. Never put a country name inside the Street_Address field.
- Birth_Country and Citizenship_Country must NEVER be auto-copied from each other.
- (Strictly) Citizenship_Country requires explicit mention in OCR text.
- Birth_Country must not be reused as fallback for missing Citizenship_Country.

## STRICT BIRTH FIELD ISOLATION
- Set `Birth_City`, `Birth_Province_or_State`, and `Birth_Country` to `null` unless explicitly labeled with terms like "Place of Birth", "Born at", or "Nato a".
- **CRITICAL:** A country name (like "USA", "United States", "Italy", "France") is NEVER a city. If a country name is detected inside a birth city context, move it to `Birth_Country` and force `Birth_City = null`.
- **CRITICAL:** NEVER use a School Name, School Address, or University location to fill birth details. If there is no explicit birth text, keep birth fields `null`.
- `Birth_Country` and `Citizenship_Country` must never be copied from each other.

BIRTH_COUNTRY ISOLATION RULE:
- Birth_Country must ONLY be extracted from the subject's own explicitly stated birth place.
- NEVER use Mother's Birth Place or Father's Birth Place to populate Birth_Country.
- If the subject's own birth country is not explicitly present, set Birth_Country = null.

STAMP & DECORATIVE TEXT IGNORE RULE:
- Ignore stamps, seals, logos, watermarks
- Never extract 'International Centre of Educational Sciences'.

## EDUCATION RULES
Extract education history into unique objects. Never infer or add fields outside the provided text.

- `School_Name`: Must be a distinct academic institution (e.g., "IMMACULATE CONCEPTION ACADEMY", "Ateneo Pontificio Regina Apostolorum").
- `Degree`: The exact degree name awarded (e.g., "Diploma", "Magisterio en Ciencias Religiosas").

Strict Target Corrections:
1. **Invalid Degree Text (Set Degree to null):** If the degree string matches variants of coursework tracking or generic years like "CICLO MAGISTERIO ANNO 3°", extract the `School_Name` but set `Degree = null`.
2. **Generic Placeholders (Remove Entire Object):** If a school row contains only a status label like "ORDINARIA" or "ORDINARIO" as the degree, do NOT extract it at all. Set both `School_Name = null` and `Degree = null`.
3. **No Faculties as Schools:** If a name contains "FACOLTA'", "FACULTY", or "DEPARTMENT", it is an academic division, not a school. Set both `School_Name = null` and `Degree = null`.
4. **Home Schooling:** Always normalize any home schooling variant to exactly `"School_Name": "HOME SCHOOLED"`.

## EDUCATION LEVEL BLOCK
- **`Education_Level` MUST ALWAYS be null.** Never populate this field under any condition in this prompt.

## RELIGIOUS & DIOCESE RULES
- `Diocese`: Extract the name of the diocese ONLY if it includes a real geographical city or place (e.g., "Diocese of Rome"). Never assign a religious movement name here.
- `Religious_Order`: Extract explicit religious organizations or movements here. 
  - **Normalization Rule:** If the text says "Movimiento de apostolado Regnum Christi" or any similar long variation, clean and shorten it to exactly **"Regnum Christi"**.
- Only populate `Religious_Status` or `Religious_Order` if explicitly written in the text.

## DATE & DOCUMENT LOCK RULES (CRITICAL)
- `Document_Type`: Must be exactly one of: [Birth Certificate, Passport, Education Certificate, Identification Proof, Religious Record, Applications Form, Other].

Separate these two rules:

1. FOR CERTIFICATES & FORMS (Birth Certificate, Education Certificate, Applications Form, Religious Record, Other):
   - `Document_Date`: Extract the date the document was signed or issued.
   - `Document_Year`: The 4-digit year of that date.
   - **Rule:** If you fill these, all `ID_Info` fields MUST be null.

2. FOR PASS-PORTS & ID CARDS ONLY (Passport, Identification Proof):
   - `Document_Issue_Date` (inside `ID_Info`): Extract the card or passport issue date ONLY.
   - **Rule:** If the document is not a Passport or ID Card, `ID_Info` and its dates MUST be null.

OCR ARTIFACT FIX RULE:
- NEVER output '105' as a year.
- Convert OCR artifacts like '105' or '/105' into '2005'.

## DOCUMENT-TYPE FIELD RESTRICTIONS (EXTENSIBLE)
The following rules forbid extracting certain fields when the detected `Document_Type` matches. If a rule applies, the listed field(s) MUST be set to `null` even if a value appears in the OCR text. Add future rules below using the same pattern.
- **Education Certificate -> Tax_Code:** If `Document_Type` is "Education Certificate", `Tax_Code` MUST be `null`. Never extract a Tax Code / Codice Fiscale / CF value from an Education Certificate, even if such a string is visible on the page.
# - <Document_Type> -> <Field>: <reason>   # (placeholder for future rules)

FINAL INSTRUCTION:
Return STRICT RAW JSON ONLY.
"""

OCR_USER_PROMPT = f"""Extract all structured data from the provided document image and return it as a single JSON object that strictly conforms to the target schema below.

Apply every rule defined in the system instructions (null enforcement, anti-hallucination, field isolation, normalization, document-type restrictions, etc.).

TARGET JSON SCHEMA:
{json.dumps(SCHEMA, ensure_ascii=False, separators=(',', ':'))}

Return STRICT RAW JSON ONLY. No markdown, no commentary, no code fences.
"""

CONSOLIDATION_SYSTEM_PROMPT = f"""
    # JSON Reconciliation Engine (Strict) — BATCH STAGE

    ## TASK
    You will receive multiple page-level JSON objects (up to {CONSOLIDATION_BATCH_SIZE} pages from the different documents of the same person).
    Merge them into **one batch-level JSON strictly matching the schema**.

    You are **NOT extracting new data**.
    You are **ONLY merging existing values**.

    ## STRICT BEHAVIOR RULE
    - Never summarize, compress, or generalize lists
    - Always treat arrays as SET UNION unless explicitly told otherwise
    - Never reduce multiple valid values into one unless a rule explicitly forces it
    - Never delete a field if it exists in any page unless explicitly forbidden

    ## OUTPUT RULES
    - Return **VALID JSON ONLY**
    - No markdown, no explanation, no extra text
    - Must match schema exactly
    - No extra fields allowed
    - No missing fields allowed
    - Preserve structure exactly
    - Never hallucinate or infer new values

    ## DOCUMENT METADATA (BATCH STAGE — IMPORTANT)
    - In THIS batch stage, `Document_Metadata` MUST be emitted as an ARRAY (list) of objects.
    - Include ONE entry per input page that has a non-empty Document_Metadata, in the original page order.
    - Each entry must preserve the original keys: `Document_Type`, `Document_Date`, `Document_Year`.
    - Do NOT collapse, deduplicate, or merge these metadata entries.
    - If a page has no Document_Metadata, skip it (do not insert empty placeholders).
    - Example shape:
      "Document_Metadata": [
        {{"Document_Type": "Passport", "Document_Date": "2019-04-12", "Document_Year": "2019"}},
        {{"Document_Type": "Education Certificate", "Document_Date": "2021-06-01", "Document_Year": "2021"}}
      ]

    ### General Rules
    - Never replace valid values with `null`
    - Ignore null or missing values
    - Each field is resolved independently unless specified

    ### Conflict Resolution
    If multiple values exist:
    - Target fields must be resolved using specific strategies below based on field type (Date/Year vs Document Type hierarchy).

    ## IDENTITY, NAME, & BIRTH DETAILS (BY DOC TYPE)
    - **Fields:** First_Name, Last_Name, Salutation, Gender, Date_Of_Birth, Birth_City, Birth_Province_or_State, Birth_Country.
    - **Rank Hierarchy:** 1. Birth Certificate -> 2. Passport -> 3. Identification Proof -> 4. Education Certificate -> 5. Applications Form -> 6. Religious Record / Other.
    - **Absolute Rule:** Resolve conflicts using the Rank Hierarchy. However, you MUST prioritize the **Birth Certificate** to fill `First_Name`, `Last_Name`, `Gender`, and all Birth Place fields if it exists in the inputs.

    ## EDUCATION RULES
    - Normalize School_Name (case-insensitive + trim + collapse spaces)
    - Use normalized School_Name as key
    - If multiple entries share same key:
        - Keep record with non-null fields
        - **Degree Consolidation Rule:** If duplicate entries exist for the same school, always prefer and keep the record with the longer, more complete, and descriptive `Degree` string (e.g., keep "Magisterio en Ciencias Religiosas" over just "Magisterio"). Drop the shorter or less descriptive duplicate.
        - Never output both duplicates
    - Remove entries where School_Name AND Degree are null
    - **Education_Level Rule:** Look at all `Degree` values in the final `Education_History` array and set `Education_Level` to match the single highest qualification found.

    ## ADDRESS & CONTACT RULES (YEAR DRIVEN)
    The following fields MUST be decided strictly by the document year:
    - Street_Address, City, State_or_Province, Zip_or_Postal_Code, Country
    - Phone, Mobile, Fax, Primary_Email

    Conflict Rule: You MUST look at `Document_Year` in each page's Document_Metadata. Select these address and contact values exclusively from the page(s) matching the latest/most recent year **within this batch**.

    ## LANGUAGE RULES (HARD CONSTRAINT)
    - Output MUST contain BOTH:
    1. Languages (array)
    2. Primary_Language (single value)

    - Languages is REQUIRED and MUST contain ALL unique non-empty languages from all pages
    - Never output empty or null Languages if any language exists in input

    - `Primary_Language`: Look at the "Primary_Language" field of each individual page. Count them. Set the final `Primary_Language` to the one that appears most frequently.
    - If there is a frequency tie, default to "English".

    - Never drop Languages for simplification

    ## BIRTH RULES
    - Birth_City must NOT contain country (move it to Birth_Country)
    - Birth_Country must be a valid country only (no states/cities/regions) especially not "Oregon".

    ## FINAL RULE
    - Output **strict JSON only**
    - `Document_Metadata` MUST be a list of objects (one per page), as described above.
    - No inferred or guessed values
    """


FINAL_CONSOLIDATION_SYSTEM_PROMPT = f"""
    # JSON Reconciliation Engine (Strict) — FINAL STAGE

    ## TASK
    You will receive multiple BATCH-level JSON objects (each already produced by merging up to {CONSOLIDATION_BATCH_SIZE} pages).
    Merge them into **one FINAL JSON strictly matching the schema**.

    You are **NOT extracting new data**.
    You are **ONLY merging existing values**.

    ## STRICT BEHAVIOR RULE
    - Never summarize, compress, or generalize lists
    - Always treat arrays as SET UNION unless explicitly told otherwise
    - Never reduce multiple valid values into one unless a rule explicitly forces it
    - Never delete a field if it exists in any batch unless explicitly forbidden

    ## OUTPUT RULES
    - Return **VALID JSON ONLY**
    - No markdown, no explanation, no extra text
    - Must match schema exactly
    - No extra fields allowed
    - No missing fields allowed
    - Preserve structure exactly
    - Never hallucinate or infer new values

    ## FORBIDDEN
    - `"Document_Metadata"` must NOT appear in the FINAL output (remove completely after using it for year-based resolution).

    ### General Rules
    - Never replace valid values with `null`
    - Ignore null or missing values
    - Each field is resolved independently unless specified

    ## INPUT NOTE
    - In each batch input, `Document_Metadata` is an ARRAY of per-page metadata objects.
    - Treat the UNION of all those arrays across batches as the full document metadata pool when resolving year-driven fields.

    ## IDENTITY, NAME, & BIRTH DETAILS (BY DOC TYPE)
    - **Fields:** First_Name, Last_Name, Salutation, Gender, Date_Of_Birth, Birth_City, Birth_Province_or_State, Birth_Country.
    - **Rank Hierarchy:** 1. Birth Certificate -> 2. Passport -> 3. Identification Proof -> 4. Education Certificate -> 5. Applications Form -> 6. Religious Record / Other.
    - **Absolute Rule:** Resolve conflicts using the Rank Hierarchy. However, you MUST prioritize the **Birth Certificate** to fill `First_Name`, `Last_Name`, `Gender`, and all Birth Place fields if it exists in the inputs.

    ## EDUCATION RULES
    - Normalize School_Name (case-insensitive + trim + collapse spaces)
    - Use normalized School_Name as key
    - If multiple entries share same key:
        - Keep record with non-null fields
        - **Degree Consolidation Rule:** Prefer and keep the record with the longer, more complete, and descriptive `Degree` string. Drop the shorter or less descriptive duplicate.
        - Never output both duplicates
    - Remove entries where School_Name AND Degree are null
    - **Education_Level Rule:** Look at all `Degree` values in the final `Education_History` array and set `Education_Level` to match the single highest qualification found.

    ## ADDRESS & CONTACT RULES (YEAR DRIVEN)
    The following fields MUST be decided strictly by document year:
    - Street_Address, City, State_or_Province, Zip_or_Postal_Code, Country
    - Phone, Mobile, Fax, Primary_Email

    Conflict Rule: Look at ALL `Document_Year` values across the metadata lists from every batch. Select these address and contact values exclusively from the batch(es) whose metadata list contains the latest/most recent year.

    ## LANGUAGE RULES (HARD CONSTRAINT)
    - Output MUST contain BOTH:
    1. Languages (array) — union of all unique non-empty languages across batches
    2. Primary_Language (single value)

    - `Primary_Language`: Choose the most frequently occurring `Primary_Language` across the batches. On tie, default to "English".

    ## BIRTH RULES
    - Birth_City must NOT contain country (move it to Birth_Country)
    - Birth_Country must be a valid country only (no states/cities/regions).

    ## FINAL RULE
    - Output **strict JSON only**
    - Do NOT include `Document_Metadata` in the final output
    - No inferred or guessed values
    """


# =====================================================
# HELPER PARSING & TOKEN UTILITIES
# =====================================================
def parse_model_json(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        return {}

    snippet = text[start:].strip()
    stack = []
    clean_index = 0
    in_string = False
    escape = False

    for i, char in enumerate(snippet):
        if char == '"' and not escape:
            in_string = not in_string
        elif escape:
            escape = False
            continue
        elif char == '\\' and in_string:
            escape = True
            continue

        if not in_string:
            if char in ['{', '[']:
                stack.append(char)
            elif char in ['}', ']']:
                if not stack:
                    break
                if (char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '['):
                    stack.pop()
                    if not stack:
                        clean_index = i + 1
                        break
        if not stack:
            clean_index = i + 1

    if stack and clean_index == 0:
        fallback_snippet = snippet
        for reverse_idx in range(len(snippet) - 1, -1, -1):
            if snippet[reverse_idx] in [',', '{', '['] and not in_string:
                fallback_snippet = snippet[:reverse_idx].strip()
                break

        repair_stack = []
        for char in fallback_snippet:
            if char in ['{', '[']:
                repair_stack.append(char)
            elif char in ['}', ']'] and repair_stack:
                repair_stack.pop()

        closure = "".join(['}' if token == '{' else ']' for token in reversed(repair_stack)])
        snippet = fallback_snippet + closure
    elif clean_index > 0:
        snippet = snippet[:clean_index]

    with suppress(Exception):
        return json.loads(snippet)

    with suppress(Exception):
        parsed = ast.literal_eval(snippet)
        if isinstance(parsed, dict):
            return parsed

    return {}


def get_zoho_access_token() -> str:
    """
    Returns a cached Zoho OAuth access token, refreshing it only when expired.
    Tokens are valid for 60 minutes; we refresh ~5 minutes early as a safety buffer.
    """
    cache = get_zoho_access_token._cache
    with cache["lock"]:
        now = time.time()
        if cache["token"] and now < cache["expires_at"]:
            return cache["token"]

        payload = {
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "refresh_token": REFRESH_TOKEN
        }
        try:
            response = requests.post(TOKEN_URL, data=payload, timeout=30)
            # Surface Zoho's actual error body (e.g. {"error":"invalid_code"})
            # instead of just the HTTP status — that's the only way to tell
            # invalid_code vs invalid_client vs wrong data center.
            if not response.ok:
                try:
                    err_body = response.json()
                except Exception:
                    err_body = response.text
                raise RuntimeError(
                    f"Zoho token endpoint returned HTTP {response.status_code} "
                    f"from {TOKEN_URL}. Response: {err_body}. "
                    f"Common causes: (1) wrong data center domain "
                    f"(set ZOHO_ACCOUNTS_DOMAIN to accounts.zoho.eu/.in/.com.au/.jp "
                    f"to match where the refresh token was issued); "
                    f"(2) refresh token revoked or regenerated; "
                    f"(3) CLIENT_ID/CLIENT_SECRET don't match the refresh token."
                )
            token_data = response.json()
            if "access_token" not in token_data:
                raise RuntimeError(f"Failed to generate token layout: {token_data}")

            # Zoho returns expires_in in seconds (typically 3600). Refresh 5 min early.
            expires_in = int(token_data.get("expires_in", 3600))
            cache["token"] = token_data["access_token"]
            cache["expires_at"] = now + max(60, expires_in - 300)
            return cache["token"]
        except Exception as e:
            raise RuntimeError(f"Zoho Authorization Gateway connection error: {str(e)}")


# Module-level token cache shared across all calls (thread-safe).
get_zoho_access_token._cache = {"token": None, "expires_at": 0.0, "lock": threading.Lock()}


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_from_image(image_path: str, access_token: str) -> dict:
    image_b64 = encode_image(image_path)
    payload = {
        "prompt": OCR_USER_PROMPT,
        "system_prompt": OCR_SYSTEM_PROMPT,
        "model": MODEL_NAME,
        "images": [image_b64],
        "top_k": 50,
        "top_p": 0.8,
        "temperature": 0.0,
        "max_tokens": 4096,
        "guided_json": SCHEMA
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
        "CATALYST-ORG": CATALYST_ORG
    }
    response = requests.post(VLM_URL, json=payload, headers=headers, timeout=300)
    result = response.json()

    # Parse the raw response to dictionary
    parsed_json = parse_model_json(result.get("response", ""))

    # --- TERMINAL PRINT ADDITION ---
    page_name = os.path.basename(image_path)
    print(f"\n{'=' * 20} TERMINAL OUTPUT: {page_name} {'=' * 20}")
    print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
    print(f"{'=' * 60}\n")
    # -------------------------------

    return parsed_json


def _call_consolidation_llm(prompt_text: str, system_prompt: str, access_token: str) -> dict:
    """
    Single LLM call to the Zoho QuickML consolidation endpoint.
    Returns the parsed JSON dict from the model response.
    """
    payload = {
        "prompt": prompt_text,
        "model": CONSOLIDATION_MODEL_NAME,
        "system_prompt": system_prompt,
        "top_p": 0.9,
        "top_k": 50,
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
        "CATALYST-ORG": CATALYST_ORG
    }
    response = requests.post(LLM_URL, json=payload, headers=headers, timeout=600)
    result = response.json()
    return parse_model_json(result.get("response", ""))


def _consolidate_batch(batch_index: int, batch_pages: list[dict], access_token: str) -> dict:
    """
    Worker: consolidate a single batch (<= CONSOLIDATION_BATCH_SIZE pages).
    Returns batch-level JSON where Document_Metadata is a list of per-page metadata.
    """
    combined_prompt = (
        f"INPUT DATA FRAGMENTS TO MERGE (Array of Page-Level JSONs for batch #{batch_index + 1}, "
        f"containing {len(batch_pages)} page(s)):\n"
        f"{json.dumps(batch_pages, ensure_ascii=False)}"
    )
    print(f"  -> Consolidating batch #{batch_index + 1} ({len(batch_pages)} pages)...")
    result = _call_consolidation_llm(combined_prompt, CONSOLIDATION_SYSTEM_PROMPT, access_token)

    # --- Log per-batch consolidation output to terminal ---
    print(f"\n{'=' * 20} BATCH #{batch_index + 1} CONSOLIDATION OUTPUT {'=' * 20}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"{'=' * 70}\n")

    return result


def consolidate_jsons(page_jsons: list[dict]) -> dict:
    """
    Two-stage parallel consolidation:
      1) Split page JSONs into batches of CONSOLIDATION_BATCH_SIZE and consolidate
         each batch in parallel via the QuickML LLM. Each batch response keeps
         `Document_Metadata` as a LIST of per-page metadata entries (logged to stdout).
      2) Make ONE final LLM call that merges all batch-level responses into the
         single final JSON (with Document_Metadata removed).
    """
    pure_data_payload = [p["data"] for p in page_jsons if "data" in p and "error" not in p["data"]]

    if not pure_data_payload:
        print("⚠️ No valid page data found to consolidate.")
        return {}

    access_token = get_zoho_access_token()

    # ---- Stage 1: build batches of CONSOLIDATION_BATCH_SIZE pages ----
    batches = [
        pure_data_payload[i:i + CONSOLIDATION_BATCH_SIZE]
        for i in range(0, len(pure_data_payload), CONSOLIDATION_BATCH_SIZE)
    ]
    print(
        f"Running batched consolidation via Zoho QuickML "
        f"({CONSOLIDATION_MODEL_NAME}): {len(batches)} batch(es) of up to "
        f"{CONSOLIDATION_BATCH_SIZE} pages each..."
    )

    # Fast path: only one batch — no need for a second final call.
    if len(batches) == 1:
        batch_result = _consolidate_batch(0, batches[0], access_token)
        if isinstance(batch_result, dict):
            batch_result.pop("Document_Metadata", None)
        return batch_result

    # Run batches in parallel (HTTP I/O bound → thread pool is appropriate).
    batch_results: list[dict] = [None] * len(batches)  # type: ignore[list-item]
    workers_count = max(1, min(len(batches), CONSOLIDATION_MAX_WORKERS))
    with ThreadPoolExecutor(max_workers=workers_count) as executor:
        future_to_idx = {
            executor.submit(_consolidate_batch, idx, batch, access_token): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                batch_results[idx] = future.result()
            except Exception as e:
                print(f"  !! Batch #{idx + 1} consolidation failed: {e}")
                batch_results[idx] = {}

    # Filter out empty failures but keep order.
    valid_batch_results = [b for b in batch_results if isinstance(b, dict) and b]

    if not valid_batch_results:
        return {}

    # ---- Stage 2: single final merge call ----
    final_prompt = (
        f"INPUT BATCH-LEVEL JSONs TO MERGE (Array of {len(valid_batch_results)} batch result(s); "
        f"each may contain a Document_Metadata LIST of per-page metadata):\n"
        f"{json.dumps(valid_batch_results, ensure_ascii=False)}"
    )
    print(f"Running FINAL consolidation merge over {len(valid_batch_results)} batch result(s)...")

    # Token may have expired during long batch runs — re-fetch via cache.
    access_token = get_zoho_access_token()
    final_result = _call_consolidation_llm(final_prompt, FINAL_CONSOLIDATION_SYSTEM_PROMPT, access_token)

    # Defensive: strip Document_Metadata if model leaked it through.
    if isinstance(final_result, dict):
        final_result.pop("Document_Metadata", None)

    # --- Log final consolidation output to terminal ---
    print(f"\n{'=' * 20} FINAL CONSOLIDATION OUTPUT {'=' * 20}")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))
    print(f"{'=' * 65}\n")

    return final_result


# =====================================================
# PDF TO IMAGES CONVERSION
# =====================================================
def pdf_to_images(pdf_path, output_folder="output_images", zoom=2):
    os.makedirs(output_folder, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    image_paths = []
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
    pdf_document.close()
    return image_paths


# =====================================================
# PROFESSIONAL UI CSS INJECTIONS
# =====================================================
st.markdown("""
<style>
    html, body, .stApp, main, .main, .main .block-container, .block-container, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {
        margin: 10px !important;
        padding: 0px !important;
        min-height: 10px !important;
    }
    .stApp { background-color: #0B0F17; }
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1600px;
    }

    #MainMenu, footer, header, [data-testid="stHeader"], [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    .header-bar {
        background: #111827;
        padding: 18px 24px;
        border-radius: 16px;
        border: 1px solid #1F2937;
        margin-bottom: 24px;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.2);
    }
            
    .header-flex { display: flex; align-items: center; gap: 14px; }
    .header-icon { font-size: 50px; line-height: 1; }
    .header-title { font-size: 22px; font-weight: 700; color: #F8FAFC; margin: 0; line-height: 1.2; }
    .header-subtitle { font-size: 13px; color: #9CA3AF; margin: 0; }

    .section-title { font-size: 1.45rem; font-weight: 650; color: #F8FAFC; margin-bottom: 1.3rem; }

    .success-box {
        background-color: rgba(16, 185, 129, 0.06);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        padding: 1.2rem;
        border-radius: 14px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }
    .status-box {
        background-color: #111827;
        border: 1px solid #1F2937;
        padding: 1rem;
        border-radius: 14px;
        color: #9CA3AF;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }
    section[data-testid="stFileUploader"] {
        border: 2px dashed #374151;
        border-radius: 16px;
        padding: 1rem;
        background: #111827;
    }
    div[data-testid="metric-container"] {
        background-color: #111827;
        border: 1px solid #1F2937;
        padding: 1rem;
        border-radius: 14px;
    }
    .metric-container-card [data-testid="stMetricLabel"] > div {
        color: white; font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important; font-weight: 500 !important;
        text-transform: uppercase !important; letter-spacing: 0.5px !important;
    }
    .metric-container-card [data-testid="stMetricValue"] > div {
        color: #FFFFFF; font-family: 'Courier New', monospace !important;
        font-size: 1.6rem !important; font-weight: 700 !important;
    }
    .file-details-heading { font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin-top: 1.5rem; margin-bottom: 1rem; }

    .stButton > button {
        width: 100%; height: 3.2rem; border-radius: 14px; border: none;
        background: #3B82F6; color: #FFFFFF; font-size: 1rem; font-weight: 600;
        transition: all 0.2s ease; margin-top: 1rem;
    }
    .stButton > button:hover { background: #2563EB; box-shadow: 0 0 12px rgba(59, 130, 246, 0.4); }

    .stDownloadButton > button {
        width: 100%; height: 3rem; border-radius: 14px; font-weight: 600;
        background: transparent; color: #3B82F6; border: 1px solid #3B82F6;
    }
    .stDownloadButton > button:hover { background: rgba(59, 130, 246, 0.1); color: #3B82F6; }

    .dashboard-card-container ::-webkit-scrollbar { width: 6px !important; height: 6px !important; }
    .dashboard-card-container ::-webkit-scrollbar-track { background: #111827 !important; border-radius: 10px !important; }
    .dashboard-card-container ::-webkit-scrollbar-thumb { background: #374151 !important; border-radius: 10px !important; }
    .dashboard-card-container ::-webkit-scrollbar-thumb:hover { background: #60A5FA !important; }

    .dashboard-card-container [data-testid="stHorizontalBlock"] { margin-bottom: 20px !important; }
    .dashboard-card-container [data-testid="column"] { display: flex; flex-direction: column; justify-content: center; }
</style>
""", unsafe_allow_html=True)

# Render Heading Banner
st.markdown(
    '<div class="header-bar">'
    '    <div class="header-flex">'
    '        <div class="header-icon">📄</div>'
    '        <div>'
    '            <div class="header-title">DOC Extractor</div>'
    '            <div class="header-subtitle">Intelligent document processing and structured JSON extraction</div>'
    '        </div>'
    '    </div>'
    '</div>',
    unsafe_allow_html=True
)

# =====================================================
# STATE CACHE MANAGEMENT
# =====================================================
for key, initial_val in [
    ("temp_dir", None), ("pdf_path", None), ("total_pages", 0),
    ("pdf_loaded", False), ("page_jsons", None), ("page_number", 0),
    ("extracted_json", None), ("preview_ready", False),
    ("extraction_start_time", None), ("extraction_elapsed", None)
]:
    if key not in st.session_state:
        st.session_state[key] = initial_val

# =====================================================
# LAYOUT STRUCTURE
# =====================================================
left_col, right_col = st.columns([1, 1], gap="large")

# =====================================================
# LEFT PANEL: DOCUMENT MANAGEMENT
# =====================================================
with left_col:
    st.markdown('<div class="dashboard-card-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Document Preview</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    # Reset session state when the uploader is cleared (file removed by user)
    if uploaded_file is None and st.session_state.get("pdf_loaded", False):
        with suppress(Exception):
            if st.session_state.temp_dir and os.path.isdir(st.session_state.temp_dir):
                shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        st.session_state.temp_dir = None
        st.session_state.pdf_path = None
        st.session_state.total_pages = 0
        st.session_state.pdf_loaded = False
        st.session_state.page_jsons = None
        st.session_state.page_number = 0
        st.session_state.extracted_json = None
        st.session_state.preview_ready = False
        st.session_state.extraction_start_time = None
        st.session_state.extraction_elapsed = None
        st.rerun()

    if uploaded_file:
        if st.session_state.temp_dir is None:
            st.session_state.temp_dir = tempfile.mkdtemp()

        if st.session_state.pdf_path is None:
            st.session_state.pdf_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
            file_bytes = uploaded_file.read()
            with open(st.session_state.pdf_path, "wb") as f:
                f.write(file_bytes)

            # Also persist a copy into the local project folder so uploaded
            # PDFs are downloadable/inspectable outside the temp directory.
            try:
                local_save_path = os.path.join(LOCAL_UPLOAD_DIR, uploaded_file.name)
                with open(local_save_path, "wb") as lf:
                    lf.write(file_bytes)
                print(f"📥 Saved uploaded file locally to: {local_save_path}")
            except Exception as e:
                print(f"⚠️ Failed to save uploaded file locally: {e}")

            st.session_state.page_jsons = None
            st.session_state.extracted_json = None
            st.session_state.page_number = 0
            st.session_state.preview_ready = False
            st.session_state.extraction_start_time = None
            st.session_state.extraction_elapsed = None

            pdf_doc = fitz.open(st.session_state.pdf_path)
            st.session_state.total_pages = len(pdf_doc)
            pdf_doc.close()

            st.session_state.pdf_loaded = True
            st.rerun()

        # ------------------------------------------------------------------
        # 10-second loader gate: show a spinner before revealing the preview.
        # The right panel also suppresses its status until preview_ready=True.
        # ------------------------------------------------------------------
        if not st.session_state.preview_ready:
            st.markdown(
                """
                <div style="display:flex; flex-direction:column; align-items:center;
                            justify-content:center; height:520px; gap:18px;">
                    <div style="
                        width:64px; height:64px; border-radius:50%;
                        border:6px solid #1F2937; border-top-color:#3B82F6;
                        animation: docspin 1s linear infinite;"></div>
                    <div style="color:#9CA3AF; font-size:0.95rem;">uploading...</div>
                </div>
                <style>
                @keyframes docspin { to { transform: rotate(360deg); } }
                </style>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(5)
            st.session_state.preview_ready = True
            st.rerun()

        # Render Page Navigation View
        pdf_doc = fitz.open(st.session_state.pdf_path)
        page = pdf_doc.load_page(st.session_state.page_number)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        pdf_doc.close()

        preview_img = os.path.join(st.session_state.temp_dir, "preview.png")
        pix.save(preview_img)

        st.image(preview_img, use_container_width=True)

        nav_col1, nav_col2, nav_col3 = st.columns([2, 3, 2])
        with nav_col1:
            if st.button("⬅ Previous"):
                if st.session_state.page_number > 0:
                    st.session_state.page_number -= 1
                    st.rerun()
        with nav_col2:
            st.markdown(
                f"<div style='display: flex; align-items: center; justify-content: center; height: 2.8rem; font-weight: 600; color: #F8FAFC; font-size: 0.95rem;'>"
                f"Page {st.session_state.page_number + 1} of {st.session_state.total_pages}</div>",
                unsafe_allow_html=True
            )
        with nav_col3:
            if st.button("Next ➡"):
                if st.session_state.page_number < st.session_state.total_pages - 1:
                    st.session_state.page_number += 1
                    st.rerun()

        st.markdown(
            f'<div class="success-box"><span style="font-weight: 600; font-size: 1rem;">✓ File Uploaded Successfully</span>'
            f'<div style="color: #9CA3AF; margin-top: 0.5rem; font-family: monospace; font-size: 0.85rem;">{uploaded_file.name}</div></div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="file-details-heading">File Details</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-container-card">', unsafe_allow_html=True)
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("File Type", "PDF")
        with m_col2:
            st.metric("Size", f"{uploaded_file.size / (1024 * 1024):.2f} MB")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FIELD LABEL MAPPING & DYNAMIC FORM RECONCILIATION
# =====================================================
FIELD_LABELS = {
    "First_Name": "First Name",
    "Last_Name": "Last Name",
    "Salutation": "Salutation",
    "Gender": "Gender",
    "Date_Of_Birth": "Date of Birth",
    "Birth_City": "Birth City",
    "Birth_Province_or_State": "Birth State / Province",
    "Birth_Country": "Birth Country",
    "Tax_Code": "Tax Code",
    "Citizenship_Country": "Citizenship Country",
    "Marital_Status": "Marital Status",
    "ID_Info": "ID Information",
    "Passport_Number": "Passport Number",
    "ID_Card_Number": "ID Card Number",
    "Document_Issue_Date": "Issue Date",
    "Document_Expiry_Date": "Expiry Date",
    "Street_Address": "Street Address",
    "City": "City",
    "State_or_Province": "State / Province",
    "Zip_or_Postal_Code": "ZIP / Postal Code",
    "Country": "Country",
    "Phone": "Phone",
    "Mobile": "Mobile",
    "Fax": "Fax",
    "Primary_Email": "Primary Email",
    "Personal_Email": "Personal Email",
    "Education_History": "Education History",
    "School_Name": "School Name", "Degree": "Degree",
    "Primary_Language": "Primary Language",
    "Languages": "Languages",
    "Education_Level": "Education Level",
    "Diocese": "Diocese", "Bishop_Email": "Bishop Email", "Bishop_Name": "Bishop Name",
    "Seminary_Name": "Seminary Name", "Seminary_Address": "Seminary Address", "Seminary_Email": "Seminary Email",
    "Religious_Status": "Religious Status", "Religious_Order": "Religious Order", "Ordination_Date": "Ordination Date"
}

SECTION_GROUPS = {
    "Personal Details": ["First_Name", "Last_Name", "Salutation", "Gender", "Date_Of_Birth"],
    "Birth Details": ["Birth_City", "Birth_Province_or_State", "Birth_Country"],
    "Identity": ["Tax_Code", "Citizenship_Country", "Marital_Status"],
    "ID / Document": ["ID_Info", "Passport_Number", "ID_Card_Number", "Document_Issue_Date", "Document_Expiry_Date"],
    "Address & Contact": ["Street_Address", "City", "State_or_Province", "Zip_or_Postal_Code", "Country", "Phone",
                          "Mobile", "Fax", "Primary_Email"],
    "Education": ["Education_History", "School_Name", "Degree", "Education_Level"],
    "Languages": ["Primary_Language", "Languages"],
    "Religious Information": ["Diocese", "Bishop_Email", "Bishop_Name", "Seminary_Name", "Seminary_Address",
                              "Seminary_Email", "Religious_Status", "Religious_Order", "Ordination_Date"]
}


def get_section_name(key):
    for section, fields in SECTION_GROUPS.items():
        if key in fields: return section
    return "Other Details"


def is_empty_field(val):
    return val in [None, "", [], {}, [""]]


def has_visible_data(val):
    """Recursively checks if a field contains any non-empty, non-null values."""
    if val in [None, "", [], {}, "None", [""]]:
        return False
    if isinstance(val, dict):
        return any(has_visible_data(v) for v in val.values())
    if isinstance(val, list):
        return any(has_visible_data(item) for item in val)
    return True


def render_dynamic_form(data, parent_key=""):
    if not isinstance(data, dict):
        return

    # Group fields by section, but ONLY if they contain actual data
    grouped = {}
    for k, v in data.items():
        if not has_visible_data(v):
            continue
        sec = get_section_name(k)
        grouped.setdefault(sec, []).append((k, v))

    # Render the valid groups
    for section_name, fields in grouped.items():
        st.markdown(
            f'<div style="margin-top: 25px; margin-bottom: 10px; font-size: 1.2rem; font-weight: 800; color: #60A5FA; border-bottom: 1px solid #2d3748; padding-bottom: 6px;">{section_name}</div>',
            unsafe_allow_html=True
        )

        with st.container(border=True):
            for key, value in fields:
                display_label = FIELD_LABELS.get(key, key.replace("_", " "))
                field_id = f"{parent_key}_{key}" if parent_key else key

                if isinstance(value, dict):
                    st.markdown(
                        f'<div style="margin-top: 15px; margin-bottom: 10px; font-size: 1.05rem; font-weight: 700; color: #60A5FA;">{display_label}</div>',
                        unsafe_allow_html=True)
                    with st.container(border=True):
                        render_dynamic_form(value, parent_key=field_id)

                elif isinstance(value, list):
                    st.markdown(
                        f'<div style="margin-top: 15px; margin-bottom: 10px; font-size: 1.05rem; font-weight: 700; color: #60A5FA;">{display_label}</div>',
                        unsafe_allow_html=True)
                    with st.container(border=True):
                        for item in value:
                            if not has_visible_data(item):
                                continue

                            # If it's a structured dictionary (like Education records), break it into formatted rows
                            if isinstance(item, dict):
                                with st.container():
                                    for sub_key, sub_value in item.items():
                                        if has_visible_data(sub_value):
                                            # Get the label (e.g., "School Name" instead of "School_Name")
                                            sub_label = FIELD_LABELS.get(sub_key, sub_key.replace("_", " "))

                                            col1, col2 = st.columns([1, 2])
                                            with col1:
                                                st.markdown(
                                                    f'<div style="padding-top: 10px; font-weight: 600; color: #E5E7EB; font-size: 0.9rem;">{sub_label}</div>',
                                                    unsafe_allow_html=True)
                                            with col2:
                                                st.markdown(
                                                    f'<div style="padding: 10px 12px; border: 1px solid #1F2937; border-radius: 10px; background: #F8FAFC; color: black; font-size: 0.95rem; margin-bottom: 8px;">{str(sub_value)}</div>',
                                                    unsafe_allow_html=True)

                                    # Add a subtle visual divider line between different schools in the list
                                    st.markdown(
                                        '<hr style="border: 0; border-top: 1px dashed #374151; margin: 15px 0;">',
                                        unsafe_allow_html=True)
                            else:
                                # For simple string arrays (like Languages)
                                st.markdown(
                                    f'<div style="padding: 10px 12px; margin-bottom: 6px; border: 1px solid #1F2937; border-radius: 12px; background: #F8FAFC; color: black;">{item}</div>',
                                    unsafe_allow_html=True)
                else:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(
                            f'<div style="padding-top: 10px; font-weight: 600; color: #E5E7EB;">{display_label}</div>',
                            unsafe_allow_html=True)
                    with col2:
                        st.markdown(
                            f'<div style="padding: 10px 12px; border: 1px solid #1F2937; border-radius: 10px; background: #F8FAFC; color: black; font-size: 0.95rem;">{str(value)}</div>',
                            unsafe_allow_html=True)


def remap_json_keys(data, mapping):
    """
    Recursively remaps dictionary keys based on a provided mapping configuration.
    """
    if isinstance(data, list):
        return [remap_json_keys(item, mapping) for item in data]

    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            # Determine the target key from the mapping dictionary
            new_key = mapping.get(k, k)

            # If the value is a dictionary or list, recurse into it
            if isinstance(v, (dict, list)):
                new_dict[new_key] = remap_json_keys(v, mapping)
            else:
                new_dict[new_key] = v
        return new_dict

    return data


# =====================================================
# RIGHT PANEL: PARALLEL AGGREGATION PIPELINE
# =====================================================
with right_col:
    st.markdown('<div class="dashboard-card-container">', unsafe_allow_html=True)

    # All right-panel work is gated by `preview_ready` so the left-side loader
    # is the only visible status during the initial 10-second delay.
    panel_active = st.session_state.get("pdf_loaded", False) and st.session_state.get("preview_ready", False)

    # ----------------------------------------------------------------------
    # AUTO-RUN STAGE 1: Per-page OCR extraction (after the 10s preview delay).
    # ----------------------------------------------------------------------
    if panel_active and st.session_state.page_jsons is None:
        try:
            # Start the extraction timer at the very beginning of stage 1.
            if st.session_state.extraction_start_time is None:
                st.session_state.extraction_start_time = time.time()
                st.session_state.extraction_elapsed = None
            ocr_progress = st.progress(0)
            ocr_status = st.empty()

            image_folder = os.path.join(st.session_state.temp_dir, "output_images")

            # Page Splitting Phase
            if not os.path.exists(image_folder) or len(os.listdir(image_folder)) == 0:
                ocr_status.markdown('<div class="status-box">Converting PDF into pages...</div>',
                                    unsafe_allow_html=True)
                pdf_to_images(st.session_state.pdf_path, image_folder)

            ocr_progress.progress(15)

            # Zoho Catalyst Cloud OCR Extraction Phase
            ocr_status.markdown('<div class="status-box">Running OCR extraction...</div>', unsafe_allow_html=True)

            image_files = sorted(
                [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            access_token = get_zoho_access_token()

            results_map = {}

            def _ocr_thread_worker(index, file_name):
                path = os.path.join(image_folder, file_name)
                extracted_data = extract_from_image(path, access_token)
                return {"page": index + 1, "file": file_name, "data": extracted_data}

            workers_count = max(1, min(len(image_files), OCR_MAX_WORKERS))
            with ThreadPoolExecutor(max_workers=workers_count) as executor:
                future_to_page = {
                    executor.submit(_ocr_thread_worker, i, file): i + 1
                    for i, file in enumerate(image_files)
                }

                completed_count = 0
                for future in as_completed(future_to_page):
                    p_num = future_to_page[future]
                    results_map[p_num] = future.result()
                    completed_count += 1

                    current_pct = 15 + int((completed_count / len(image_files)) * 80)
                    ocr_progress.progress(min(95, current_pct))
                    ocr_status.markdown(
                        f'<div class="status-box">Extracted Page {p_num} of {len(image_files)}...</div>',
                        unsafe_allow_html=True)

            st.session_state.page_jsons = [results_map[i + 1] for i in range(len(image_files))]
            ocr_progress.progress(100)
            ocr_status.empty()
            ocr_progress.empty()
            st.rerun()

        except Exception as e:
            st.error(f"OCR Extraction Aborted: {str(e)}")

    st.markdown('<div class="section-title">Candidate Details</div>', unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # AUTO-RUN STAGE 2: Consolidation runs automatically as soon as page JSONs
    # are ready and no final result exists yet. No button click required.
    # ----------------------------------------------------------------------
    if panel_active and st.session_state.page_jsons is not None and (
            st.session_state.extracted_json is None or "error" in (st.session_state.extracted_json or {})):
        try:
            progress_bar = st.progress(0)
            status_box = st.empty()

            progress_bar.progress(25)
            status_box.markdown('<div class="status-box">Running layout consolidation...</div>',
                                unsafe_allow_html=True)

            st.session_state.extracted_json = consolidate_jsons(st.session_state.page_jsons)

            progress_bar.progress(100)

            # Stop the extraction timer once the final JSON has been produced.
            if (
                st.session_state.extraction_start_time is not None
                and isinstance(st.session_state.extracted_json, dict)
                and "error" not in st.session_state.extracted_json
            ):
                st.session_state.extraction_elapsed = (
                    time.time() - st.session_state.extraction_start_time
                )

            if "error" in st.session_state.extracted_json:
                status_box.markdown(
                    f'<div class="status-box" style="color: #EF4444;">Pipeline Execution Error: {st.session_state.extracted_json["error"]}</div>',
                    unsafe_allow_html=True)
            else:
                status_box.markdown('<div class="success-box">Document processed successfully </div>',
                                    unsafe_allow_html=True)
                st.rerun()

        except Exception as e:
            st.error(f"Extraction Pipeline Aborted: {str(e)}")

    # ====================================================================
    # RENDER SECTION: UI Forms, Dynamic Mapping, and View Toggle
    # ====================================================================
    if st.session_state.extracted_json and "error" not in st.session_state.extracted_json:

        # Display the total extraction time (OCR + consolidation pipeline).
        elapsed = st.session_state.get("extraction_elapsed")
        if SHOW_EXTRACTION_TIME and elapsed is not None:
            mins, secs = divmod(elapsed, 60)
            if mins >= 1:
                time_str = f"{int(mins)}m {secs:.1f}s"
            else:
                time_str = f"{secs:.2f}s"
            st.markdown(
                f'<div class="success-box" style="display:flex; align-items:center; gap:10px;">'
                f'<span style="font-size:1.1rem;">⏱️</span>'
                f'<span><b>Total Extraction Time:</b> '
                f'<span style="font-family: \'Courier New\', monospace; color:#F8FAFC;">{time_str}</span></span>'
                f'</div>',
                unsafe_allow_html=True
            )

        OUTPUT_JSON_MAPPING = {
            "First_Name": "First_Name", "Last_Name": "Last_Name", "Salutation": "Salutation", "Gender": "Genere",
            "Date_Of_Birth": "Date_of_Birth", "Birth_City": "Citt_di_nascita",
            "Birth_Province_or_State": "Regione_di_nascitta", "Birth_Country": "Nazione_di_nascitta",
            "Tax_Code": "CF", "Citizenship_Country": "Cittadinanza", "Marital_Status": "Stato_civile",
            "Passport_Number": "Passaporto", "ID_Card_Number": "Carta_identit",
            "Document_Issue_Date": "Data_di _emissione", "Document_Expiry_Date": "Data_di _scadenza",
            "Street_Address": "Mailing_Street", "City": "Mailing_City", "State_or_Province": "Mailing_State",
            "Zip_or_Postal_Code": "Mailing_Zip", "Country": "Mailing_Country",
            "Phone": "Phone", "Mobile": "Mobile", "Fax": "Fax", "Primary_Email": "Email", "Personal_Email": "Email_personale",
            "Education_History": "Storia_istruzione", "School_Name": "School_Name", "Degree": "Degree",
            "Primary_Language": "Lingua", "Languages": "Lingue", "Education_Level": "Livello",
            "Diocese": "Diocesi", "Bishop_Email": "Email_del_Vescovo", "Bishop_Name": "S_E_R_Mons",
            "Seminary_Name": "Nome_del_seminario", "Seminary_Address": "Indirizzo_del_seminario",
            "Seminary_Email": "E_mail_del_seminario",
            "Religious_Status": "Stato_religioso", "Religious_Order": "Congregazione",
            "Ordination_Date": "Data_ordinazione_sacerdotale"
        }

        remapped_json = remap_json_keys(st.session_state.extracted_json, OUTPUT_JSON_MAPPING)

        if isinstance(remapped_json, dict) and "ID_Info" in remapped_json:
            id_data = remapped_json.pop("ID_Info", {})
            if isinstance(id_data, dict):
                for id_k, id_v in id_data.items():
                    mapped_id_k = OUTPUT_JSON_MAPPING.get(id_k, id_k)
                    remapped_json[mapped_id_k] = id_v

        output_json_str = json.dumps(remapped_json, indent=2, ensure_ascii=False)

        # View Mode Toggle Switch
        view_mode_json = st.toggle("View Raw Output JSON Schema", value=False)
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

        # Dynamic Switcher View Display Frame Container
        form_container = st.container(height=650, border=False)
        with form_container:
            if view_mode_json:
                st.code(output_json_str, language="json")
            else:
                render_dynamic_form(st.session_state.extracted_json)

        # Target Action Output Stream Download Button
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="Download Final JSON",
            data=output_json_str,
            file_name="verified_output.json",
            mime="application/json"
        )
    else:
        # Context-aware guide text for the right panel.
        if not st.session_state.get("pdf_loaded", False):
            # Initial state — no file uploaded yet.
            st.markdown(
                '<div class="status-box" style="margin-top: 10px;">Upload a PDF document to preview pages and extract structured fields into a form.</div>',
                unsafe_allow_html=True)
        elif not panel_active:
            # File uploaded but still inside the 10-second preview loader window.
            # Keep the right panel intentionally quiet — the left-side spinner is the only status.
            pass

