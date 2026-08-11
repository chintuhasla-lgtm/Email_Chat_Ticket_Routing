"""02_text_cleaning.py
Load tickets, clean ticket body text, show examples, and save cleaned output.

Creates a `clean_text` column by removing quoted replies, signatures,
and boilerplate disclaimers, lowercasing, stripping punctuation, and
removing stopwords (from NLTK).
"""
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords


def ensure_stopwords():
    """Download NLTK stopwords corpus if not already present."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


def clean_ticket_text(text: str) -> str:
    """Clean ticket text and return a normalized token string.

    Steps:
    - drop quoted-reply lines starting with '>'
    - truncate at reply headers like 'On ... wrote:'
    - truncate at common legal disclaimer phrases
    - truncate at signature markers like 'Regards,', 'Thanks,'
    - remove contact lines (emails, phone numbers)
    - lowercase, remove punctuation, and strip English stopwords
    """
    if not isinstance(text, str):
        return ""

    # Normalize newlines and strip outer whitespace
    txt = text.replace("\r\n", "\n").strip()

    # Remove lines that are quoted (start with '>')
    lines = [ln for ln in txt.split("\n") if not ln.strip().startswith(">")]
    txt = "\n".join(lines).strip()

    # Truncate at common reply headers like: "On Tue, Jan 1, 2020, John Doe <...> wrote:"
    m = re.search(r"\bOn\s.+?wrote:", txt, flags=re.IGNORECASE | re.DOTALL)
    if m:
        txt = txt[: m.start()].strip()

    # Truncate at common disclaimer / confidentiality markers
    disclaimer_patterns = [
        r"This email and any attachments",
        r"If you are not the intended recipient",
        r"Confidential",
        r"legal disclaimer",
        r"This message .* confidential",
    ]
    for pat in disclaimer_patterns:
        dm = re.search(pat, txt, flags=re.IGNORECASE)
        if dm:
            txt = txt[: dm.start()].strip()
            break

    # Truncate at signature markers (common sign-offs)
    sig_markers = ["Regards,", "Thanks,", "Best,", "Sincerely,", "Kind regards,"]
    for marker in sig_markers:
        idx = txt.find(marker)
        if idx != -1:
            txt = txt[:idx].strip()
            break

    # Remove remaining contact lines that look like emails or phone numbers
    kept_lines = []
    for ln in txt.split("\n"):
        if re.search(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}", ln):
            continue
        if re.search(r"(\+?\d[\d\-\s\(\)]{6,}\d)", ln):
            continue
        kept_lines.append(ln.strip())
    txt = " ".join([l for l in kept_lines if l])

    # Lowercase and remove punctuation (keep word characters and whitespace)
    txt = txt.lower()
    txt = re.sub(r"[^\w\s]", " ", txt)

    # Tokenize and remove stopwords
    words = re.findall(r"\b\w+\b", txt)
    sw = set(stopwords.words("english"))
    words = [w for w in words if w not in sw]

    return " ".join(words)


if __name__ == "__main__":
    # Ensure NLTK stopwords are available
    ensure_stopwords()

    # Load dataset
    df = pd.read_csv("data/customer_support_tickets.csv")

    # Apply cleaning to the Body_Text column and create `clean_text`
    df["clean_text"] = df["Body_Text"].apply(clean_ticket_text)

    # Print raw vs cleaned text for 3 sample rows so differences are visible
    samples = df.dropna(subset=["Body_Text"]).sample(n=3, random_state=1)
    for _, row in samples.iterrows():
        print("---")
        print(f"Ticket_ID: {row.get('Ticket_ID', '')}")
        print("Raw:\n", row.get("Body_Text", ""))
        print("\nCleaned:\n", row.get("clean_text", ""))
        print()

    # Save the cleaned dataframe
    df.to_csv("data/cleaned_tickets.csv", index=False)
    print("Saved cleaned tickets to data/cleaned_tickets.csv")
