"""Check SFT-paper/main_ijra against the IJRA rules in PROMPT_PAPER7_IJRA_OPUS.md.

Reads the compiled PDF, the section sources, and the .bbl/.bib. Prints one line per rule.

Usage: python -m scripts.audit_manuscript
"""
import re
import subprocess
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2] / "SFT-paper"
SECTIONS = ["01_introduction", "02_related_work", "03_method", "04_results", "05_conclusion"]
FORBIDDEN = [r"\bledger\b", r"\bartefact", r"\bartifact", r"\bevidence\b", r"\bpilot\b",
             r"\b[FPM][0-9]+b?\b", r"\\pb\{", r"PLACEHOLDER", r"\bTODO\b", r"\bnovel\b",
             r"\bstate-of-the-art\b", r"\bgroundbreaking\b", r"\bremarkabl", r"\bsignificantly\b",
             r"\bwe\b", r"\bour\b", r"\bdelve"]


def texcount(paths):
    out = subprocess.run(["texcount", "-1", "-sum", "-merge", *map(str, paths)],
                         capture_output=True, text=True).stdout.strip()
    return int(out.split()[0]) if out else -1


def strip_comments(s):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in s.splitlines())


def main():
    pdf = PAPER / "main_ijra.pdf"
    pages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True,
                                                             text=True).stdout).group(1))
    body = texcount([PAPER / "sections" / f"{s}.tex" for s in SECTIONS])
    abstract_src = strip_comments((PAPER / "sections" / "00_abstract.tex").read_text())
    abstract = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.,%]*", re.sub(r"\\[a-z]+", " ", abstract_src)))
    total_pdf = len(subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout.split())
    kw = [k.strip() for k in strip_comments((PAPER / "sections" / "00_keywords.tex").read_text()).split(r"\sep")]
    kw = [k for k in kw if k]
    intro_src = re.sub(r"\\section\{[^}]*\}", "", strip_comments((PAPER / "sections" / "01_introduction.tex").read_text()))
    intro_pars = [p for p in intro_src.split("\n\n") if p.strip()]

    bbl = (PAPER / "main_ijra.bbl").read_text()
    cited = re.findall(r"\\bibitem\{([^}]+)\}", bbl)
    cited = [c for c in cited if c != "BSTcontrol"]
    bib = (PAPER / "ref.bib").read_text()
    entries = {m.group(2): (m.group(1).lower(), m.group(3)) for m in
               re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", bib, re.S)}
    recent_journal = []
    for c in cited:
        kind, fields = entries.get(c, ("?", ""))
        year = re.search(r"year\s*=\s*\{?(\d{4})", fields)
        if kind == "article" and year and int(year.group(1)) >= 2021:
            recent_journal.append(c)
    self_cites = [c for c in cited if "hilmi" in c.lower()]

    abstract_cite = "\\cite" in abstract_src
    hits = {}
    for s in SECTIONS + ["00_abstract", "00_keywords"]:
        txt = strip_comments((PAPER / "sections" / f"{s}.tex").read_text())
        for pat in FORBIDDEN:
            for m in re.finditer(pat, txt, re.I if pat.startswith(r"\b") and pat[2].islower() else 0):
                hits.setdefault(pat, []).append(f"{s}:{txt[:m.start()].count(chr(10)) + 1}")
    decisions = []
    for f in sorted((PAPER / "sections").glob("*.tex")) + [PAPER / "main_ijra.tex"]:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if "DECISION FOR ARDI" in line:
                decisions.append(f"{f.name}:{i}")

    def row(ok, rule, val):
        print(f"[{'PASS' if ok else 'FAIL'}] {rule}: {val}")

    row(pages == 8, "exactly 8 pages", pages)
    row(body + abstract <= 4000, "words, body + abstract <= 4,000", f"{body} + {abstract} (all PDF text {total_pdf})")
    row(100 <= abstract <= 200, "abstract 100-200 words", abstract)
    row(not abstract_cite, "abstract without citations", "ok" if not abstract_cite else "has \\cite")
    row(4 <= len(kw) <= 7 and kw == sorted(kw, key=str.lower), "keywords 4-7, alphabetical", kw)
    row(3 <= len(intro_pars) <= 7, "introduction 3-7 paragraphs", len(intro_pars))
    row(len(cited) >= 25, "references >= 25", len(cited))
    row(len(recent_journal) >= 15, "recent (>=2021) journal articles >= 15", f"{len(recent_journal)} {recent_journal}")
    row(len(self_cites) <= min(5, 0.2 * len(cited)), "self-citations <= 5 and <= 20%", self_cites)
    row(not hits, "no forbidden / draft / promotional terms", hits or "none")
    print(f"[INFO] DECISION FOR ARDI markers: {decisions}")


if __name__ == "__main__":
    main()
