#!/usr/bin/env python3
"""
lit_review.py — Literature review: search, rank, download.

Supports arXiv and PubMed sources.

Usage:
    python3 lit_review.py search "query" --dir /path [--source arxiv|pubmed] [--max-papers 50] [--abstracts-only]
    python3 lit_review.py cleanup --dir /path --keep 50 "overall topic description"
    python3 lit_review.py list --dir /path
"""

import json
import re
import shutil
import sys
import time
import ssl
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────

EMBEDDING_MODEL = "text-embedding-3-small"
ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_DELAY = 3  # seconds between arXiv requests

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_DELAY = 1  # seconds between PubMed requests (3 req/s without key)

# SSL context for HTTPS requests (macOS Python sometimes lacks system certs)
_ssl_ctx = ssl.create_default_context()
try:
    import certifi
    _ssl_ctx.load_verify_locations(certifi.where())
except ImportError:
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = ssl.CERT_NONE


# ── Slugify ────────────────────────────────────────────────────────────

def slugify(text, max_len=50):
    """Convert search terms to a filesystem-safe slug."""
    s = text.lower().strip()
    s = re.sub(r'[^a-z0-9\s-]', '', s)
    s = re.sub(r'[\s]+', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s[:max_len]


# ── ArXiv API ──────────────────────────────────────────────────────────

def search_arxiv(query_terms, max_results=200):
    """Search arXiv API and return list of paper metadata dicts."""
    candidates = []
    page_size = 100
    pages = (max_results + page_size - 1) // page_size

    for page in range(pages):
        start = page * page_size
        remaining = max_results - start
        fetch = min(page_size, remaining)

        params = urllib.request.quote(query_terms)
        url = f"{ARXIV_API}?search_query=all:{params}&start={start}&max_results={fetch}&sortBy=relevance"

        if page > 0:
            print(f"  Waiting {ARXIV_DELAY}s before next API page...")
            time.sleep(ARXIV_DELAY)

        print(f"  Fetching arXiv results {start+1}-{start+fetch}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'lit-review-tool/1.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_data = resp.read().decode('utf-8')
        except Exception as e:
            print(f"  Warning: arXiv API request failed: {e}")
            break

        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        entries = root.findall('atom:entry', ns)
        if not entries:
            break

        for entry in entries:
            title = entry.find('atom:title', ns)
            abstract = entry.find('atom:summary', ns)
            published = entry.find('atom:published', ns)
            arxiv_id_el = entry.find('atom:id', ns)

            if title is None or abstract is None or arxiv_id_el is None:
                continue

            # Extract arXiv ID from URL like http://arxiv.org/abs/2401.12345v1
            raw_id = arxiv_id_el.text.strip()
            arxiv_id = raw_id.split('/abs/')[-1]
            # Strip version suffix for PDF URL
            base_id = re.sub(r'v\d+$', '', arxiv_id)

            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text.strip())

            # Find PDF link
            pdf_url = f"https://arxiv.org/pdf/{base_id}"
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'pdf':
                    pdf_url = link.get('href')
                    break

            candidates.append({
                'arxiv_id': arxiv_id,
                'base_id': base_id,
                'title': ' '.join(title.text.strip().split()),
                'abstract': ' '.join(abstract.text.strip().split()),
                'authors': authors,
                'published': published.text.strip() if published is not None else '',
                'pdf_url': pdf_url,
            })

        if len(entries) < fetch:
            break  # No more results

    return candidates


# ── PubMed API ────────────────────────────────────────────────────────

def search_pubmed(query_terms, max_results=200):
    """Search PubMed via NCBI E-utilities and return list of paper metadata dicts."""
    # Step 1: esearch to get PMIDs
    params = urllib.parse.urlencode({
        'db': 'pubmed',
        'term': query_terms,
        'retmax': max_results,
        'sort': 'relevance',
        'retmode': 'json',
    })
    url = f"{PUBMED_ESEARCH}?{params}"

    print(f"  Searching PubMed for up to {max_results} results...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'lit-review-tool/1.0'})
        with urllib.request.urlopen(req, timeout=30, context=_ssl_ctx) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"  Warning: PubMed esearch failed: {e}")
        return []

    id_list = data.get('esearchresult', {}).get('idlist', [])
    if not id_list:
        print("  No results found on PubMed.")
        return []

    print(f"  Found {len(id_list)} PMIDs, fetching metadata...")

    # Step 2: efetch in batches to get full metadata (XML)
    candidates = []
    batch_size = 100
    for batch_start in range(0, len(id_list), batch_size):
        batch_ids = id_list[batch_start:batch_start + batch_size]

        if batch_start > 0:
            print(f"  Waiting {PUBMED_DELAY}s before next batch...")
            time.sleep(PUBMED_DELAY)

        fetch_params = urllib.parse.urlencode({
            'db': 'pubmed',
            'id': ','.join(batch_ids),
            'rettype': 'xml',
            'retmode': 'xml',
        })
        fetch_url = f"{PUBMED_EFETCH}?{fetch_params}"

        try:
            req = urllib.request.Request(fetch_url, headers={'User-Agent': 'lit-review-tool/1.0'})
            with urllib.request.urlopen(req, timeout=60, context=_ssl_ctx) as resp:
                xml_data = resp.read().decode('utf-8')
        except Exception as e:
            print(f"  Warning: PubMed efetch failed: {e}")
            continue

        root = ET.fromstring(xml_data)
        for article_el in root.findall('.//PubmedArticle'):
            paper = _parse_pubmed_article(article_el)
            if paper:
                candidates.append(paper)

    return candidates


def _parse_pubmed_article(article_el):
    """Parse a single PubmedArticle XML element into a paper metadata dict."""
    medline = article_el.find('.//MedlineCitation')
    if medline is None:
        return None

    pmid_el = medline.find('PMID')
    if pmid_el is None:
        return None
    pmid = pmid_el.text.strip()

    article = medline.find('Article')
    if article is None:
        return None

    # Title
    title_el = article.find('ArticleTitle')
    title = ' '.join(title_el.itertext()).strip() if title_el is not None else ''
    if not title:
        return None

    # Abstract
    abstract_el = article.find('Abstract')
    if abstract_el is not None:
        abstract_parts = []
        for text_el in abstract_el.findall('AbstractText'):
            label = text_el.get('Label', '')
            text = ' '.join(text_el.itertext()).strip()
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = ' '.join(abstract_parts)
    else:
        abstract = ''

    # Authors
    authors = []
    author_list = article.find('AuthorList')
    if author_list is not None:
        for author_el in author_list.findall('Author'):
            last = author_el.find('LastName')
            fore = author_el.find('ForeName')
            if last is not None:
                name = last.text.strip()
                if fore is not None:
                    name = f"{fore.text.strip()} {name}"
                authors.append(name)

    # Published date
    pub_date = article.find('.//PubDate')
    published = ''
    if pub_date is not None:
        year = pub_date.find('Year')
        month = pub_date.find('Month')
        if year is not None:
            published = year.text.strip()
            if month is not None:
                published = f"{published}-{month.text.strip()}"

    # PMC ID for free full-text PDF
    pmc_id = None
    article_ids = article_el.find('.//PubmedData/ArticleIdList')
    if article_ids is not None:
        for aid in article_ids.findall('ArticleId'):
            if aid.get('IdType') == 'pmc':
                pmc_id = aid.text.strip()
                break

    # DOI
    doi = None
    if article_ids is not None:
        for aid in article_ids.findall('ArticleId'):
            if aid.get('IdType') == 'doi':
                doi = aid.text.strip()
                break

    # PDF URL (only available for PMC open-access papers)
    pdf_url = None
    if pmc_id:
        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"

    return {
        'pmid': pmid,
        'base_id': f"pmid_{pmid}",
        'title': title,
        'abstract': abstract,
        'authors': authors,
        'published': published,
        'pdf_url': pdf_url,
        'pmc_id': pmc_id,
        'doi': doi,
        'source': 'pubmed',
    }


# ── Embedding-based ranking ───────────────────────────────────────────

def rank_by_relevance(query_text, candidates, openai_client, top_n=50):
    """Rank candidates by cosine similarity to query embedding."""
    if not candidates:
        return []

    # Filter out candidates with empty abstracts (OpenAI rejects empty strings)
    valid_candidates = [c for c in candidates if c.get('abstract', '').strip()]
    if not valid_candidates:
        return candidates[:top_n]

    texts = [query_text] + [c['abstract'] for c in valid_candidates]
    # Truncate to stay within token limits
    texts = [t[:8000] for t in texts]

    print(f"  Embedding {len(texts)} texts for relevance ranking...")
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        all_embeddings.extend([e.embedding for e in response.data])

    query_emb = all_embeddings[0]
    abstract_embs = all_embeddings[1:]

    # Cosine similarity (embeddings are already normalized by OpenAI)
    scored = []
    for i, emb in enumerate(abstract_embs):
        dot = sum(a * b for a, b in zip(query_emb, emb))
        scored.append((dot, i))

    scored.sort(reverse=True)
    selected = []
    for sim, idx in scored[:top_n]:
        paper = dict(valid_candidates[idx])
        paper['similarity'] = round(sim, 4)
        selected.append(paper)

    return selected


# ── Search command ─────────────────────────────────────────────────────

def cmd_search(query_terms, output_dir, max_papers=50, abstracts_only=False, source='arxiv',
               skip_ranking=False):
    """Full search workflow: search → rank → download."""
    # Set up output directory
    review_dir = Path(output_dir)
    papers_dir = review_dir / "papers"

    review_dir.mkdir(parents=True, exist_ok=True)
    papers_dir.mkdir(exist_ok=True)

    print(f"=== Literature Review: {query_terms} ===")
    print(f"Source: {source}")
    print(f"Directory: {review_dir}\n")

    print(f"[Phase 1/3] Searching {source}...")
    if source == 'pubmed':
        candidates = search_pubmed(query_terms, max_results=200)
    else:
        candidates = search_arxiv(query_terms, max_results=200)
    print(f"  Found {len(candidates)} candidates\n")

    if not candidates:
        print(f"No results found on {source}. Try different search terms.")
        return

    candidates_path = review_dir / "candidates.json"
    with open(candidates_path, 'w') as f:
        json.dump(candidates, f, indent=2)

    if skip_ranking:
        print("[Phase 2/3] Skipping embedding ranking (using source ordering)")
        selected = candidates[:max_papers]
    else:
        openai_client = OpenAI()
        print("[Phase 2/3] Ranking by abstract relevance...")
        selected = rank_by_relevance(query_terms, candidates, openai_client, top_n=max_papers)
    print(f"  Selected top {len(selected)} papers")
    if selected and selected[0].get('similarity'):
        print(f"  Similarity range: {selected[-1]['similarity']:.3f} - {selected[0]['similarity']:.3f}")
    print()

    # Merge with existing selected_papers.json (accumulate across runs)
    selected_path = review_dir / "selected_papers.json"
    existing = {}
    if selected_path.exists():
        with open(selected_path) as f:
            for p in json.load(f):
                existing[p.get('base_id', p.get('arxiv_id'))] = p

    for p in selected:
        key = p.get('base_id', p.get('arxiv_id'))
        if key not in existing or p.get('similarity', 0) > existing[key].get('similarity', 0):
            existing[key] = p

    merged = list(existing.values())
    merged.sort(key=lambda p: p.get('similarity', 0), reverse=True)
    with open(selected_path, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"  Accumulated {len(merged)} unique papers in selected_papers.json")

    # Print top papers
    print("  Top 10 papers:")
    for i, p in enumerate(selected[:10]):
        sim = p.get('similarity')
        prefix = f"[{sim:.3f}] " if sim is not None else ""
        print(f"    {i+1}. {prefix}{p['title'][:80]}")
    print()

    # Download PDFs (unless abstracts-only)
    downloaded = 0
    skipped_no_pdf = 0
    delay = PUBMED_DELAY if source == 'pubmed' else ARXIV_DELAY
    if abstracts_only:
        print("[Phase 3/3] Skipping PDF download (--abstracts-only)")
    else:
        downloadable = [p for p in selected if p.get('pdf_url')]
        no_pdf = len(selected) - len(downloadable)
        if no_pdf > 0:
            print(f"  Note: {no_pdf} papers have no free PDF (behind paywall)")
        print(f"[Phase 3/3] Downloading {len(downloadable)} PDFs...")
        for i, paper in enumerate(downloadable):
            pdf_path = papers_dir / f"{paper['base_id'].replace('/', '_')}.pdf"
            if pdf_path.exists():
                downloaded += 1
                continue

            print(f"  [{i+1}/{len(downloadable)}] Downloading {paper['base_id']}...")
            if i > 0:
                time.sleep(delay)

            try:
                req = urllib.request.Request(
                    paper['pdf_url'],
                    headers={'User-Agent': 'lit-review-tool/1.0'}
                )
                with urllib.request.urlopen(req, timeout=60, context=_ssl_ctx) as resp:
                    content = resp.read()
                    # Verify we got a PDF (PMC sometimes returns HTML)
                    if content[:5] == b'%PDF-' or len(content) > 1000:
                        with open(pdf_path, 'wb') as f:
                            f.write(content)
                        downloaded += 1
                    else:
                        print(f"    Warning: response was not a PDF, skipping")
                        skipped_no_pdf += 1
            except Exception as e:
                print(f"    Warning: download failed: {e}")

        print(f"  Downloaded {downloaded}/{len(downloadable)} PDFs\n")

    # Summary
    total_size = sum(
        f.stat().st_size for f in review_dir.rglob('*') if f.is_file()
    )
    size_mb = total_size / (1024 * 1024)

    today = date.today().isoformat()
    manifest = {
        'query': query_terms,
        'source': source,
        'created': today,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'candidates_count': len(candidates),
        'selected_count': len(selected),
        'downloaded_count': downloaded,
        'abstracts_only': abstracts_only,
        'disk_usage_mb': round(size_mb, 1),
        'directory': str(review_dir),
    }
    with open(review_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Literature review complete!")
    print(f"  Query:      {query_terms}")
    print(f"  Papers:     {len(selected)} selected from {len(candidates)} candidates")
    print(f"  Downloaded: {downloaded}")
    print(f"  Disk usage: {size_mb:.1f} MB")
    print(f"  Directory:  {review_dir}")
    print(f"\nTo build a knowledge graph:")
    print(f"  /knowledge-graph {review_dir}")
    print(f"{'='*60}")


# ── Cleanup command ────────────────────────────────────────────────────

def cmd_cleanup(review_dir_str, query_terms, keep=50):
    """Re-rank all papers against a unified query, keep top N, delete the rest."""
    review_dir = Path(review_dir_str)
    papers_dir = review_dir / "papers"
    selected_path = review_dir / "selected_papers.json"

    if not selected_path.exists():
        print(f"Error: no selected_papers.json in {review_dir}")
        sys.exit(1)

    with open(selected_path) as f:
        all_papers = json.load(f)

    if not all_papers:
        print("No papers to clean up.")
        return

    print(f"=== Cleanup: re-ranking {len(all_papers)} papers ===")
    print(f"  Query: {query_terms}")
    print(f"  Keep:  {keep}\n")

    # Re-rank all papers against the unified query
    openai_client = OpenAI()
    ranked = rank_by_relevance(query_terms, all_papers, openai_client, top_n=len(all_papers))

    # Split into keep vs remove
    kept = ranked[:keep]

    # Delete PDFs not in the kept set (single pass)
    kept_stems = {p.get('base_id', '').replace('/', '_') for p in kept}
    deleted_count = 0
    deleted_bytes = 0
    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        if pdf_path.stem not in kept_stems:
            size = pdf_path.stat().st_size
            pdf_path.unlink()
            deleted_count += 1
            deleted_bytes += size

    # Write updated selected_papers.json with only kept papers
    with open(selected_path, 'w') as f:
        json.dump(kept, f, indent=2)

    # Update manifest
    total_size = sum(
        f.stat().st_size for f in review_dir.rglob('*') if f.is_file()
    )
    size_mb = total_size / (1024 * 1024)
    saved_mb = deleted_bytes / (1024 * 1024)

    manifest_path = review_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}
    manifest['selected_count'] = len(kept)
    manifest['cleanup_query'] = query_terms
    manifest['cleanup_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    manifest['disk_usage_mb'] = round(size_mb, 1)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Cleanup complete!")
    print(f"  Kept:    {len(kept)} papers")
    print(f"  Removed: {deleted_count} PDFs ({saved_mb:.1f} MB freed)")
    print(f"  Disk:    {size_mb:.1f} MB remaining")
    if kept:
        print(f"  Similarity range: {kept[-1].get('similarity', 0):.3f} - {kept[0].get('similarity', 0):.3f}")
    print(f"\nTo build a knowledge graph:")
    print(f"  /knowledge-graph {review_dir}")
    print(f"{'='*60}")


# ── List command ───────────────────────────────────────────────────────

def cmd_list(scan_dir):
    """List all existing literature reviews in a directory."""
    scan_path = Path(scan_dir)
    reviews = sorted(scan_path.glob("*/manifest.json"))

    if not reviews:
        print(f"No literature reviews found in {scan_path}")
        return

    print(f"{'Name':<45} {'Papers':>6} {'Size':>8}  Query")
    print("-" * 80)

    for manifest_path in reviews:
        with open(manifest_path) as f:
            m = json.load(f)
        name = manifest_path.parent.name
        mode = " [abs]" if m.get('abstracts_only') else ""
        print(f"{name:<45} {m.get('selected_count', '?'):>6} {m.get('disk_usage_mb', 0):>6.1f}MB  {m.get('query', '?')}{mode}")


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args or args[0] in ('--help', '-h'):
        print(__doc__)
        return

    cmd = args[0]

    if cmd == 'search':
        # Parse: search "query" --dir <path> [--source arxiv|pubmed] [--max-papers N] [--abstracts-only]
        query_parts = []
        max_papers = 50
        abstracts_only = False
        skip_ranking = False
        output_dir = None
        source = 'arxiv'
        i = 1
        while i < len(args):
            if args[i] == '--max-papers' and i + 1 < len(args):
                max_papers = int(args[i + 1])
                i += 2
            elif args[i] == '--dir' and i + 1 < len(args):
                output_dir = args[i + 1]
                i += 2
            elif args[i] == '--source' and i + 1 < len(args):
                source = args[i + 1].lower()
                if source not in ('arxiv', 'pubmed'):
                    print(f"Error: --source must be 'arxiv' or 'pubmed', got '{source}'")
                    sys.exit(1)
                i += 2
            elif args[i] == '--abstracts-only':
                abstracts_only = True
                i += 1
            elif args[i] == '--skip-ranking':
                skip_ranking = True
                i += 1
            else:
                query_parts.append(args[i])
                i += 1
        query_text = ' '.join(query_parts)
        if not query_text:
            print("Error: provide search terms after 'search'")
            sys.exit(1)
        if not output_dir:
            print("Error: --dir <path> is required")
            sys.exit(1)
        cmd_search(query_text, output_dir=output_dir, max_papers=max_papers,
                   abstracts_only=abstracts_only, source=source, skip_ranking=skip_ranking)

    elif cmd == 'cleanup':
        # Parse: cleanup --dir <path> --keep N "query terms"
        output_dir = None
        keep = 50
        query_parts = []
        i = 1
        while i < len(args):
            if args[i] == '--dir' and i + 1 < len(args):
                output_dir = args[i + 1]
                i += 2
            elif args[i] == '--keep' and i + 1 < len(args):
                keep = int(args[i + 1])
                i += 2
            else:
                query_parts.append(args[i])
                i += 1
        if not output_dir:
            print("Error: --dir <path> is required")
            sys.exit(1)
        query_text = ' '.join(query_parts)
        if not query_text:
            print("Error: provide a query describing the overall topic")
            sys.exit(1)
        cmd_cleanup(output_dir, query_text, keep=keep)

    elif cmd == 'list':
        # Parse: list --dir <path>
        scan_dir = None
        i = 1
        while i < len(args):
            if args[i] == '--dir' and i + 1 < len(args):
                scan_dir = args[i + 1]
                i += 2
            else:
                i += 1
        if not scan_dir:
            print("Error: --dir <path> is required (directory containing review subdirectories)")
            sys.exit(1)
        cmd_list(scan_dir)

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: search, cleanup, list")
        sys.exit(1)


if __name__ == '__main__':
    main()
