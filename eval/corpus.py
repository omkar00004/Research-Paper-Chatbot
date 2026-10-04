"""PDF -> clean text once (cached), offset-tracking chunker, passage sampling, quote location.

Parsing and cleaning reuse the production code (backend.ingestion.ingest_pdf, backend.chunking._clean_text);
the splitter uses the production separators so default-config chunk texts are identical (see parity_check).
"""
import bisect, re, unicodedata
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from eval.common import CACHE, CORPUS_DIR, LEGACY_DIRS, read_json, sha, write_json
from backend.ingestion import ingest_pdf
from backend.chunking import _clean_text

# must match backend/chunking.py::chunk_documents
SEPS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
DOCS_DIR = CACHE / "docs"


def pdf_list(corpus="main"):
    if corpus == "main":
        return sorted(CORPUS_DIR.glob("*.pdf"))
    out = []
    for p in LEGACY_DIRS:
        out += sorted(p.glob("*.pdf")) if p.is_dir() else [p]
    return sorted(out, key=lambda p: p.name)


def load_docs(corpus="main"):
    """{doc_name: {"text": str, "pages": [[start, end, page_no], ...]}} with the cleaned text cached on disk."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    docs = {}
    for pdf in pdf_list(corpus):
        f = DOCS_DIR / (pdf.stem + "." + sha(pdf.stat().st_size, pdf.stat().st_mtime_ns)[:8] + ".json")
        d = read_json(f)
        if d is None:
            parts, pages, pos = [], [], 0
            for page in ingest_pdf(pdf):
                t = _clean_text(page.text)
                if not t: continue
                pages.append([pos, pos + len(t), page.metadata["page"]]); parts.append(t); pos += len(t) + 2
            d = {"text": "\n\n".join(parts), "pages": pages}
            write_json(f, d)
        docs[pdf.name] = d
    return docs


# ---------------------------------------------------------------- chunking
def chunk_doc(name, doc, size, overlap):
    """Chunks of one doc with (start, end) offsets into doc['text']. Returns (chunks, n_offset_fallbacks)."""
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, length_function=len, separators=SEPS, add_start_index=True)
    text, out, fb = doc["text"], [], 0
    for p0, p1, pno in doc["pages"]:
        ptxt = text[p0:p1]
        for piece in sp.create_documents([ptxt]):
            raw = piece.page_content; t = raw.strip()
            if not t: continue
            s = p0 + piece.metadata["start_index"] + (len(raw) - len(raw.lstrip()))
            if text[s:s + len(t)] != t:
                s = text.find(t, p0); fb += 1
                if s < 0: continue
            out.append({"id": f"{name}::{s}", "doc": name, "start": s, "end": s + len(t), "text": t, "hash": sha(t),
                        "metadata": {"source": name, "page": pno, "section": "Unknown"}})
    return out, fb


def chunk_corpus(docs, size, overlap):
    chunks, fb = [], 0
    for name, d in docs.items():
        c, f = chunk_doc(name, d, size, overlap); chunks += c; fb += f
    return chunks, fb


def parity_check(docs_main_or_legacy):
    """Eval chunker == production chunk_documents (texts, order) at the production default config."""
    from backend.chunking import chunk_documents
    from backend.config import CHUNK_SIZE, CHUNK_OVERLAP
    bad = 0
    for pdf in pdf_list("main")[:6]:
        prod = [c.text for c in chunk_documents(ingest_pdf(pdf))]
        mine = [c["text"] for c in chunk_doc(pdf.name, docs_main_or_legacy[pdf.name], CHUNK_SIZE, CHUNK_OVERLAP)[0]]
        bad += prod != mine
    return bad


# ---------------------------------------------------------------- quote location
_QMAP = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-"}
_norm_cache = {}

def _norm_with_map(text, key):
    if key in _norm_cache: return _norm_cache[key]
    out, mp, last_sp = [], [], True
    for i, c in enumerate(text):
        if c.isspace():
            if not last_sp: out.append(" "); mp.append(i); last_sp = True
            continue
        n = _QMAP.get(c) or unicodedata.normalize("NFKC", c).lower()
        for ch in n: out.append(ch); mp.append(i)
        last_sp = False
    _norm_cache[key] = ("".join(out), mp)
    return _norm_cache[key]

def _norm(s):
    o, last = [], True
    for c in s:
        if c.isspace():
            if not last: o.append(" "); last = True
            continue
        o.append(_QMAP.get(c) or unicodedata.normalize("NFKC", c).lower()); last = False
    return "".join(o).strip()

def locate_quote(name, text, quote, lo, hi):
    """Whitespace/case-insensitive search for `quote`; the match must lie inside text[lo:hi]. Returns (start, end) in original text or None."""
    q = _norm(quote)
    if len(q) < 25: return None
    nt, mp = _norm_with_map(text, name)
    for m in re.finditer(re.escape(q), nt):
        s, e = mp[m.start()], mp[m.end() - 1] + 1
        if lo <= s and e <= hi + 1: return s, e
    return None

def corpus_has_term(docs, term):
    t = _norm(term)
    return any(t in _norm_with_map(d["text"], n)[0] for n, d in docs.items())


# ---------------------------------------------------------------- passages
HEAD = re.compile(r"^(?:(?:\d+(?:\.\d+)*\.?|[A-Z]\.|Appendix(?: [A-Z])?)\s+[A-Z][^\n]{2,80}|Abstract|ABSTRACT|Introduction|INTRODUCTION|Conclusions?|Related Work|References|REFERENCES|Limitations|Acknowledg\w+|Discussion)\s*$", re.M)
SEC_RULES = [("refs", r"reference|bibliograph"), ("ack", r"acknowledg|ethic|limitation|appendix"), ("intro", r"abstract|introduc|background|motivation"),
             ("related", r"related|prior work|literature"), ("method", r"method|approach|model|framework|system|architecture|task|dataset|data |annotation|pipeline|design|formulat|prelim"),
             ("results", r"experiment|result|evaluat|analys|ablation|setup|baseline|finding|performance|study"), ("discussion", r"discuss|conclusion|future|summary")]

def section_type(head):
    h = head.lower()
    for t, pat in SEC_RULES:
        if re.search(pat, h): return t
    return "other"

BAD = re.compile(r"(?i)copyright|permission to|licen[sc]e|all rights reserved|creative commons|acknowledg|table of contents|author contributions|funding|conflict of interest|corresponding author|©|proceedings of the|isbn|universit|institute|department of|@\w+\.\w+|\.{5,}|https?://|arxiv|doi:|appendix")

def passage_ok(t):
    if not 450 <= len(t) <= 1300: return False
    nonsp = re.sub(r"\s", "", t)
    if len(re.findall(r"[A-Za-z]", nonsp)) / len(nonsp) < 0.72: return False
    if len(re.findall(r"\d", nonsp)) / len(nonsp) > 0.08: return False
    if len(re.findall(r"[.!?](?:\s|$)", t)) < 2: return False
    if len(re.findall(r"\b(?:19|20)\d\d\b", t)) >= 4 or len(re.findall(r"et al", t)) >= 3 or BAD.search(t): return False
    lines = [l for l in t.split("\n") if l.strip()]
    if lines and sum(map(len, lines)) / len(lines) < 25: return False
    if sum(len(l.strip()) <= 6 for l in lines) >= 2 or re.search(r'(?m)^\s*(?:Figure|Fig\.|Table|Algorithm)\s+\d', t): return False
    return True

def eligible_passages(name, doc):
    """Sentence-aligned windows (450-1300 chars) from body text, with a section type; references/boilerplate/tables filtered."""
    text = doc["text"]
    heads = [(m.start(), section_type(m.group(0).strip())) for m in HEAD.finditer(text)]
    hpos = [h[0] for h in heads]
    ref_start = next((p for p, t in reversed(heads) if t == "refs" and p > 0.45 * len(text)), len(text) + 1)
    bounds = [0] + [m.end() for m in re.finditer(r"[.!?](?=\s+[A-Z(\[])\s+", text)] + [len(text)]
    out, i = [], 0
    while i < len(bounds) - 1:
        j = i + 1
        while j < len(bounds) - 1 and bounds[j] - bounds[i] < 450: j += 1
        s, e = bounds[i], bounds[j]
        t = text[s:e].strip()
        k = bisect.bisect_right(hpos, s) - 1
        stype = heads[k][1] if k >= 0 else "front"
        if s < ref_start and e <= ref_start and stype not in ("refs", "ack") and passage_ok(t):
            ls = s + (len(text[s:e]) - len(text[s:e].lstrip()))
            out.append({"id": f"{name}::{ls}", "doc": name, "start": ls, "end": ls + len(t), "section": stype})
        i = j
    return out
