import os
import re
import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# ✅ Load SpaCy **only once** at API startup
print("Loading SpaCy model... (this happens only once)")
nlp = spacy.load('fr_core_news_md')  # Preloaded globally

# File paths for dictionary files
EN_FR_FILE = "en-fr-enwiktionary.txt"
FR_EN_FILE = "fr-en-enwiktionary.txt"

# Initialize the lemmatizer once
lemmatizer = WordNetLemmatizer()

# ✅ Load dictionary files **once** at API startup
def load_lines(file_path):
    """Load all lines from a file into a list (stripping newline characters)."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]
    return []

EN_FR_LINES = load_lines(EN_FR_FILE)
FR_EN_LINES = load_lines(FR_EN_FILE)

# ✅ Prebuild indices for fast lookup
def build_index(lines):
    """Build an index mapping a normalized key (the phrase before '{') to a list of lines."""
    index = {}
    for line in lines:
        m = re.match(r"^(.*?)\s*\{", line, re.IGNORECASE)
        if m:
            key = m.group(1).strip().lower()
            index.setdefault(key, []).append(line)
    return index

en_fr_index = build_index(EN_FR_LINES)
fr_en_index = build_index(FR_EN_LINES)

# ✅ FastAPI instance
app = FastAPI()

# ✅ Request model for input validation
class WordRequest(BaseModel):
    word: str


# --- Utility Functions ---
def clean_input(word):
    """Clean input by removing 'to ' (English verbs) and stripping French articles."""
    word = word.strip().lower()
    FRENCH_ARTICLES = ("le ", "la ", "les ", "l'")
    if word.startswith("to "):
        word = word[3:]
    for article in FRENCH_ARTICLES:
        if word.startswith(article):
            word = word[len(article):]
    return word

def lemmatize_word_english(word):
    """Lemmatize an English word as both a noun and a verb."""
    noun_lemma = lemmatizer.lemmatize(word, wordnet.NOUN)
    verb_lemma = lemmatizer.lemmatize(word, wordnet.VERB)
    return {word, noun_lemma, verb_lemma}

def lemmatize_word_french(word):
    """Lemmatize a French word using preloaded SpaCy model."""
    doc = nlp(word)
    return {word, doc[0].lemma_ if doc else word}

def extract_see_reference(line):
    """Extract the SEE reference phrase from a dictionary entry."""
    see_match = re.search(r"SEE:\s*(.*?)\s*::", line, re.IGNORECASE)
    return see_match.group(1).strip() if see_match else None

def remove_phonetics(text):
    """Remove phonetic markings enclosed in forward slashes."""
    return re.sub(r'/[^/]+/', '', text)

def remove_extra_spaces(text):
    """Remove extra spaces after closing curly brace '}'."""
    return re.sub(r'\}\s+', '} ', text)


# --- Core Search Functions ---
def find_lines_in_index(index, search_word, visited=None):
    """Look up lines from the pre-built index that match the search_word."""
    if visited is None:
        visited = set()
    results = []
    key = search_word.lower()
    if key in visited:
        return results  # Prevent recursion loops
    visited.add(key)
    if key in index:
        for line in index[key]:
            see_ref = extract_see_reference(line)
            if see_ref:
                recursed = find_lines_in_index(index, see_ref, visited)
                if recursed and len(recursed) <= 2:
                    results.extend(recursed)
                elif not recursed:
                    results.append(line)
            else:
                results.append(line)
    return results

def find_lines_after_colon_from_lines(lines, search_phrase):
    """Find lines where the exact search_phrase appears after '::'."""
    matching_lines = []
    for line in lines:
        if "::" in line:
            after_text = line.split("::", 1)[1]
            if search_phrase.lower() in after_text.lower():
                matching_lines.append(line)
    return matching_lines

def translation_has_letters(line):
    """Check if the part after '::' in a dictionary line contains letters."""
    if "::" in line:
        return bool(re.search(r"[a-zA-Z]", line.split("::", 1)[1].strip()))
    return True


# --- Translation API Endpoint ---
@app.post("/translate-word/")
async def translate_word_api(request: WordRequest):
    """Translate a word between English and French."""
    cleaned_word = clean_input(request.word)

    # Determine input language
    input_lang = "EN" if cleaned_word in en_fr_index else "FR" if cleaned_word in fr_en_index else "EN"

    results = []

    # Direct matches: English -> French
    direct_matches_en = find_lines_in_index(en_fr_index, cleaned_word)
    for line in direct_matches_en:
        if translation_has_letters(line):
            results.append(f"EN -> FR: {line}")

    # Direct matches: French -> English
    direct_matches_fr = find_lines_in_index(fr_en_index, cleaned_word)
    for line in direct_matches_fr:
        if translation_has_letters(line):
            results.append(f"FR -> EN: {line}")

    # Lemmatized matches
    for lemma in lemmatize_word_english(cleaned_word):
        lemma_matches_en = find_lines_in_index(en_fr_index, lemma)
        for line in lemma_matches_en:
            if translation_has_letters(line):
                res_line = f"EN -> FR: {line}"
                if res_line not in results:
                    results.append(res_line)

    for lemma in lemmatize_word_french(cleaned_word):
        lemma_matches_fr = find_lines_in_index(fr_en_index, lemma)
        for line in lemma_matches_fr:
            if translation_has_letters(line):
                res_line = f"FR -> EN: {line}"
                if res_line not in results:
                    results.append(res_line)

    # Reverse search if no matches found
    if not results:
        reverse_results = []
        rev_label = "EN -> FR:" if input_lang == "EN" else "FR -> EN:"

        reverse_matches_en = find_lines_after_colon_from_lines(EN_FR_LINES, cleaned_word)
        for line in reverse_matches_en:
            if translation_has_letters(line):
                parts = line.split("::", 1)
                swapped = parts[1].strip() + " :: " + parts[0].strip() if len(parts) == 2 else line
                reverse_results.append(f"{rev_label} {swapped}")

        reverse_matches_fr = find_lines_after_colon_from_lines(FR_EN_LINES, cleaned_word)
        for line in reverse_matches_fr:
            if translation_has_letters(line):
                parts = line.split("::", 1)
                swapped = parts[1].strip() + " :: " + parts[0].strip() if len(parts) == 2 else line
                reverse_results.append(f"{rev_label} {swapped}")

        results.extend(reverse_results)

    # Deduplicate & clean results
    seen = set()
    deduped_results = []
    for res in results:
        res_clean = remove_phonetics(remove_extra_spaces(res)).strip()
        if res_clean not in seen:
            seen.add(res_clean)
            deduped_results.append(res_clean)

    return {
        "input_word": request.word,
        "cleaned_word": cleaned_word,
        "translations": deduped_results if deduped_results else ["Translation not found."]
    }

# ✅ Run the API server using:
# uvicorn translator_api:app --host 0.0.0.0 --port 8000 --reload
