"""
SHL Catalog Scraper — Individual Test Solutions (type=1)
Run: python scripts/scrape_catalog.py
Outputs: data/catalog.json

Requires: pip install requests beautifulsoup4
Note: SHL catalog uses server-side pagination. This script handles pages
      at intervals of 12. If JavaScript rendering is needed, install Playwright:
      pip install playwright && playwright install chromium
"""
import json
import time
import sys
from pathlib import Path

# Try requests + BS4 first (fast, no headless browser needed)
try:
    import requests
    from bs4 import BeautifulSoup
    USE_PLAYWRIGHT = False
except ImportError:
    USE_PLAYWRIGHT = True

BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def scrape_with_requests() -> list:
    """Scrape catalog pages using requests + BeautifulSoup."""
    assessments = []
    start = 0
    page_size = 12

    session = requests.Session()
    session.headers.update(HEADERS)

    while True:
        url = f"{BASE_URL}?start={start}&type=1&action_doFilteringForm=Search"
        print(f"Fetching: {url}")
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"  ERROR: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # SHL catalog rows are typically in a table or div grid
        rows = soup.select("tr.product-catalogue__row") or \
               soup.select(".catalogue__row") or \
               soup.select("[data-course-code]")

        if not rows:
            # Try generic link pattern for catalog items
            rows = soup.select("a[href*='/product-catalog/view/']")

        if not rows:
            print(f"  No items found at start={start}. Stopping.")
            break

        for row in rows:
            try:
                # Extract name and URL
                link = row if row.name == "a" else row.find("a", href=lambda h: h and "/product-catalog/view/" in h)
                if not link:
                    continue
                name = link.get_text(strip=True)
                href = link["href"]
                if not href.startswith("http"):
                    href = "https://www.shl.com" + href

                # Try to get type codes from row
                type_cell = row.select_one(".product-catalogue__type, .type-code, [data-type]")
                test_type = [type_cell.get_text(strip=True)] if type_cell else ["K"]

                assessment = {
                    "name": name,
                    "url": href,
                    "test_type": test_type,
                    "description": "",
                    "job_levels": [],
                    "remote_testing": True,
                    "adaptive_irt": False,
                    "duration_minutes": None,
                    "languages": ["English"],
                    "keywords": []
                }
                assessments.append(assessment)
                print(f"  Found: {name}")
            except Exception as e:
                print(f"  Parse error: {e}")
                continue

        # Check if there's a next page
        next_btn = soup.select_one("a.pagination__next, a[rel='next']")
        if not next_btn:
            start += page_size
            # Stop after reasonable number of pages
            if start > 500:
                break
            time.sleep(0.5)
        else:
            break

    return assessments


def scrape_with_playwright() -> list:
    """Scrape catalog using headless browser (for JavaScript-rendered content)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright not installed. Run: pip install playwright && playwright install chromium")
        return []

    assessments = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers(HEADERS)

        start = 0
        while True:
            url = f"{BASE_URL}?start={start}&type=1&action_doFilteringForm=Search"
            print(f"Playwright fetching: {url}")
            page.goto(url, wait_until="networkidle", timeout=30000)
            time.sleep(1)

            # Find all product links
            links = page.query_selector_all("a[href*='/product-catalog/view/']")
            if not links:
                print(f"  No items at start={start}. Stopping.")
                break

            for link in links:
                try:
                    name = link.inner_text().strip()
                    href = link.get_attribute("href")
                    if href and not href.startswith("http"):
                        href = "https://www.shl.com" + href

                    # Try to get type from parent row
                    parent = link.evaluate("el => el.closest('tr') ? el.closest('tr').innerText : ''")

                    assessment = {
                        "name": name,
                        "url": href,
                        "test_type": ["K"],  # Default; refine from detail page
                        "description": "",
                        "job_levels": [],
                        "remote_testing": True,
                        "adaptive_irt": False,
                        "duration_minutes": None,
                        "languages": ["English"],
                        "keywords": []
                    }
                    if name and href:
                        assessments.append(assessment)
                        print(f"  Found: {name}")
                except Exception as e:
                    print(f"  Parse error: {e}")

            start += 12
            if start > 500:
                break

        browser.close()

    return assessments


def enrich_from_detail_pages(assessments: list, session=None) -> list:
    """Visit each assessment's detail page to get description and metadata."""
    if session is None:
        import requests as req
        session = req.Session()
        session.headers.update(HEADERS)

    for i, item in enumerate(assessments):
        print(f"  Enriching {i+1}/{len(assessments)}: {item['name']}")
        try:
            resp = session.get(item["url"], timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")

            # Description
            desc_el = soup.select_one(".product-catalogue__description, .content-block p, meta[name='description']")
            if desc_el:
                item["description"] = desc_el.get("content", "") or desc_el.get_text(strip=True)

            # Job levels
            level_els = soup.select(".job-level, [data-job-level]")
            if level_els:
                item["job_levels"] = [el.get_text(strip=True) for el in level_els]

            time.sleep(0.3)
        except Exception as e:
            print(f"    Error enriching {item['name']}: {e}")

    return assessments


def main():
    output_path = Path(__file__).parent.parent / "data" / "catalog.json"

    print("=== SHL Catalog Scraper ===")
    print(f"Target: {BASE_URL}")
    print(f"Output: {output_path}\n")

    if USE_PLAYWRIGHT:
        print("Using Playwright (headless browser)...")
        assessments = scrape_with_playwright()
    else:
        print("Using requests + BeautifulSoup...")
        assessments = scrape_with_requests()

    if not assessments:
        print("\nNo assessments scraped. The catalog may require JavaScript rendering.")
        print("Try: pip install playwright && playwright install chromium && python scripts/scrape_catalog.py")
        print("\nFalling back to pre-built catalog.json (already present).")
        sys.exit(0)

    # Remove duplicates by URL
    seen_urls = set()
    unique = []
    for item in assessments:
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            unique.append(item)

    print(f"\nScraped {len(unique)} unique assessments.")

    # Optionally enrich (uncomment for fuller data)
    # print("Enriching with detail page data...")
    # unique = enrich_from_detail_pages(unique)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
