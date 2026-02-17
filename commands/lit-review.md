---
allowed-tools: Bash, Read, Task
description: Search arXiv for papers on a topic, download PDFs to a directory
user-invocable: true
---

# Literature Review: $ARGUMENTS

Search arXiv for papers on a topic, rank by relevance, and download PDFs. Uses an iterative multi-phase strategy: conversation mining, diversified search, abstract snowball, citation chasing with Semantic Scholar, and cleanup.

## Instructions

### Step 0 — Parse arguments

The first argument is the **output directory** (required). Remaining arguments are search context.
- Example: `/lit-review /tmp/cylindric-review cylindric partitions positivity --max-papers 25`
- If no directory is provided, tell the user: "Please provide a directory as the first argument."
- Default max-papers per query: 20 (we run many queries, so keep each moderate)
- If the user says "abstracts only", add `--abstracts-only` to all search commands

### Step 1 — Mine conversation context

Before generating any queries, systematically extract from the **current conversation history** and your **memory files**:

- **Author names**: anyone mentioned by name (e.g., "Warnaar", "Corteel", "Borodin")
- **Paper titles or arXiv IDs**: any specific papers referenced — download these directly (see "Direct download" in Reference below)
- **Conjectures/theorems**: named results (e.g., "Conjecture 2.7", "Rogers-Ramanujan identities")
- **Techniques/methods**: named approaches (e.g., "Bailey lemma", "crystal bases", "0-Hecke algebra")
- **Key objects**: mathematical structures central to the discussion (e.g., "cylindric partitions", "quasi-symmetric functions")
- **Adjacent fields**: related areas that came up (e.g., "representation theory", "statistical mechanics")

Write a brief list of what you found — this becomes your seed vocabulary for query generation. Tell the user what you extracted so they can correct/add to it before you proceed.

### Step 2 — Round 1: Diversified search (5-8 queries)

Using the seed vocabulary from Step 1, generate queries covering different angles. Use **all** of the following strategies that apply:

| Strategy | arXiv query | Why |
|----------|-------------|-----|
| **Direct** | `"cylindric partitions positivity"` | User's exact topic phrase |
| **Synonyms** | `"periodic plane partitions"` | Alternative terminology for the same concept |
| **Adjacent techniques** | `"Bailey lemma q-series"` | Methods known to apply to this area |
| **Broader framing** | `"generating function positivity combinatorics"` | Same problem at a higher level |
| **Prolific author (full corpus)** | `au:Warnaar` | Dedicate entire queries to key authors with **no topic constraint**. Use `au:` prefix for precision. The embedding ranker will surface their relevant work, including papers that use completely different terminology. Do this for 1-3 authors who are central to the topic. |
| **Title search** | `ti:"cylindric partitions"` | When you want exact phrase matching in paper titles |
| **Survey/review papers** | `"survey" OR "review" cylindric partitions` | A single good survey contains hundreds of organized references — someone else's completed lit review. Always include one survey query. |

Run each query sequentially:
```bash
python3 <repo>/bin/lit_review.py search "<query>" --dir <dir> --max-papers <N> [--abstracts-only]
```
where `<repo>` is the path to this repository (e.g., `~/git/math-research-tools`).

Set timeout to 600000 (10 minutes). The script deduplicates — PDFs already in `papers/` are skipped.

**Diminishing returns**: After each query, note how many *new* papers were downloaded (the script reports this). If a query adds 0-2 new papers, that search angle is saturated — don't run similar queries.

### Step 3 — Check in with user

Pause and report what Round 1 found:
- How many papers total, which queries were most/least productive
- Any surprising themes or directions emerging
- Ask: "I'm seeing [X unexpected direction]. Should I pursue that, or focus on [Y]?"
- Ask if there are authors/papers/angles they want to add before snowball rounds

This avoids spending 10+ minutes chasing a direction the user doesn't care about.

### Step 4 — Snowball: Scan abstracts for new leads

Read the accumulated results using the Read tool on `<dir>/selected_papers.json`. Analyze the abstracts looking for:

- **New author names** that appear in multiple high-similarity papers but weren't in your Step 1 seed list
- **Co-author network** — if Author A wrote a relevant paper with co-author B, B likely works on adjacent problems. Search for B independently.
- **Unfamiliar terms** — jargon or technique names you haven't seen in the conversation, suggesting a different research community working on similar problems
- **Frequently co-occurring concepts** — pairs of topics that keep appearing together, suggesting a connection worth exploring
- **Surprising connections** — unexpected fields or applications

**Quality/drift check**: Before generating follow-up queries, scan the abstracts for off-topic drift. If a significant portion of results are from an unrelated field (e.g., "cylindric" matching papers about cylindrical lenses in optics), note this and adjust queries to be more specific. You can use `ti:` prefix to restrict to title matches, or add disambiguating terms.

**Category cross-listing**: Check which arXiv categories your top papers appear in (visible in arXiv metadata). If papers are cross-listed across categories (e.g., `math.CO` and `math.QA`, or `math.CO` and `math-ph`), the other category may harbour a different community working on the same structures with different language. Run a category-scoped query to explore: `cat:math.QA AND "partitions"`.

Generate 2-4 follow-up queries based on what you found:

| Strategy | Example |
|----------|---------|
| **Prolific new author** | `au:DiscoveredAuthor` (full corpus, let ranker filter) |
| **Co-author hop** | `au:CoauthorLastName` for frequent collaborators of key authors |
| **New terminology** | Terms from abstracts not in your original queries |
| **Cross-field bridge** | Unexpected connections to other areas |
| **Adjacent category** | `cat:math.QA AND "keyword"` for cross-listed categories |

Run these the same way as Round 1. **Stop early** if queries are returning mostly duplicates (diminishing returns).

### Step 5 — Citation chasing: PDFs + Semantic Scholar

The most powerful discovery step. Four sub-strategies:

#### 5a — Read introductions and related work sections

Pick the **3-5 highest-relevance PDFs** from `<dir>/papers/` and read their **first 3-5 pages** (introduction + related work) using the Read tool. These sections are expert-curated mini literature reviews — they explain not just *what* to read but *why* it matters and *how* it connects. Extract:

- Key papers described as foundational/motivating
- Techniques described as "the standard approach" or "state of the art"
- Open problems or conjectures mentioned alongside the user's topic
- Connections to other fields the authors consider important

#### 5b — Read reference lists

Read the **last few pages** of the same PDFs (the bibliography). Look for:
- Papers cited by multiple of your top papers (convergence = importance)
- Recent papers (last 2-3 years) by the same authors
- Titles that directly address the user's question

For papers with known arXiv IDs, download directly (see Reference below). For others, search arXiv by author + key title words.

#### 5c — Semantic Scholar forward + backward citations

For the 3-5 most important papers (from any round), use the Semantic Scholar API to find both who they cite and who cites them:

```bash
# Find a paper's Semantic Scholar ID by arXiv ID
curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:2401.12345?fields=title,paperId"

# Backward citations: what does this paper cite? (references)
curl -s "https://api.semanticscholar.org/graph/v1/paper/<paperId>/references?fields=title,authors,externalIds,year&limit=100"

# Forward citations: who cites this paper? (citations)
curl -s "https://api.semanticscholar.org/graph/v1/paper/<paperId>/citations?fields=title,authors,externalIds,year&limit=100"
```

From the results:
- **Forward citations** (who cites X): Find recent papers that build on foundational work — these represent the state of the art. Sort by year descending, look for papers from the last 3 years.
- **Backward citations** (what does X cite): Find the foundational papers that the best papers in your corpus all rely on.
- For papers with arXiv IDs in `externalIds`, download directly (see Reference below).
- Otherwise, search arXiv by author + title keywords.

**Rate limiting**: Semantic Scholar allows 100 requests/5 minutes without an API key. Add 1-2s delays between calls.

#### 5d — Semantic Scholar recommendations

For the 2-3 most central papers in your corpus, use the recommendations endpoint to find similar papers you may have missed entirely:

```bash
curl -s "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/<paperId>?fields=title,authors,externalIds,year&limit=20"
```

This uses Semantic Scholar's ML model to find papers similar in content, even if they use completely different terminology or aren't cited by/citing the seed paper. Download any promising results with arXiv IDs.

### Step 6 — Cleanup: re-rank and trim

After all search rounds are done, the directory may have 100-200+ PDFs from all the different queries. Run cleanup to re-rank everything against a single unified query and keep only the top N:

```bash
python3 <repo>/bin/lit_review.py cleanup --dir <dir> --keep 50 "overall topic description that captures all facets"
```

The cleanup command:
- Re-embeds all accumulated paper abstracts against one unified query
- Ranks by similarity, keeps the top `--keep` papers
- **Deletes PDFs** outside the cut from `papers/`
- Updates `selected_papers.json` to contain only the kept papers

The query string should be a comprehensive description of the overall topic (not just one narrow angle). Combine the user's original request with key concepts discovered during the search.

### Step 7 — Report results

After cleanup, report:
- **Corpus size**: papers kept / total found, disk usage
- **Discovery summary**: what each phase uncovered
  - Conversation mining: "Extracted N authors, M techniques from context"
  - Round 1: "Ran N queries covering [topics]"
  - Snowball: "Discovered authors X, Y; co-authors A, B; new terms C, D"
  - Citation chasing: "Read introductions of top papers; S2 forward citations of [key paper] revealed P1, P2; recommendations surfaced P3"
  - Cleanup: "Re-ranked N papers, kept top 50, freed X MB"
- **Key finds**: 2-3 most surprising or important discoveries
- **Directory**: where results are stored
- **Next step**: "To build a knowledge graph: `/knowledge-graph <dir>`"

## Reference

### arXiv query syntax

The arXiv API supports field-specific prefixes for more precise searches:
- `all:query` — search all fields (default, what the script uses)
- `au:LastName` — search by author name only
- `ti:keyword` — search in title only
- `abs:keyword` — search in abstract only
- `cat:math.CO` — restrict to an arXiv category
- Combine with AND/OR: `au:Warnaar AND ti:cylindric`, `cat:math.QA AND "partitions"`

The script passes the query string directly to the arXiv API `search_query` parameter. Use these prefixes for precision, especially for author queries (`au:` avoids matching papers that merely cite the author).

### Direct download by arXiv ID

When you know a paper's arXiv ID (from a reference list, Semantic Scholar, or the conversation), download it directly instead of searching:

```bash
# Download a paper by arXiv ID
curl -sL -o <dir>/papers/<arxiv_id_with_underscores>.pdf "https://arxiv.org/pdf/<arxiv_id>"

# Examples:
curl -sL -o <dir>/papers/2401.12345.pdf "https://arxiv.org/pdf/2401.12345"
curl -sL -o <dir>/papers/math_0123456.pdf "https://arxiv.org/pdf/math/0123456"
```

Use `-sL` (silent + follow redirects). This skips the search/rank/download pipeline entirely — use it for papers you already know you want.

**Note**: Papers downloaded this way won't have metadata in `selected_papers.json`. If you want them included in cleanup ranking, add a search that would find them, or accept that cleanup may delete them (they'll still survive if they rank in the top N).

## Notes
- ArXiv requests are rate-limited (3s between requests) — this is intentional
- Semantic Scholar allows 100 requests/5 minutes without API key — add 1-2s delays between calls
- The `--dir` flag is **required** — the script has no default directory
- `selected_papers.json` **accumulates** across runs (merges by arxiv_id, keeps highest similarity)
- Multiple searches to the same directory accumulate papers (deduplication by PDF filename)
- Be generous during search rounds (`--max-papers 20-30`), then let cleanup trim to the final corpus
- The `--abstracts-only` flag skips PDF downloads (~96% disk savings) but disables Step 5
- **Quick mode**: if the user asks for a quick/small review, do only Steps 0-2 and skip cleanup
- The Read tool can read PDFs natively — use it for introductions and reference sections in Step 5
- When downloading PDFs found via Semantic Scholar, save them to `<dir>/papers/` to keep everything in one place
- **Always run cleanup** (Step 6) unless the user explicitly asks to keep everything
- **Monitor diminishing returns**: if 2+ consecutive queries add fewer than 3 new papers, stop that strategy and move on
