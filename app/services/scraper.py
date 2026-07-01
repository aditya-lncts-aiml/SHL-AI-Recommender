from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import Page, sync_playwright


logger = logging.getLogger(__name__)
BASE_URL = "https://www.shl.com"
START_URLS = [
    "https://www.shl.com/products/product-catalog/",
    "https://www.shl.com/products/assessments/",
    "https://www.shl.com/products/assessments/personality-assessment/",
    "https://www.shl.com/products/assessments/cognitive-assessment/",
    "https://www.shl.com/products/assessments/skills-assessment/",
]
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "catalog.json"


@dataclass(frozen=True)
class ScrapedAssessment:
    name: str
    description: str
    test_type: str
    category: str
    duration: str
    languages: list[str]
    url: str


class SHLCatalogScraper:
    def __init__(self, start_urls: list[str] | None = None, output_path: Path = OUTPUT_PATH) -> None:
        self.start_urls = start_urls or START_URLS
        self.output_path = output_path

    def scrape(self, max_pages: int = 250, headless: bool = True) -> list[ScrapedAssessment]:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=headless)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(30000)
            links = self._discover_links(page, max_pages=max_pages)
            assessments = [self._scrape_assessment(page, link) for link in links]
            browser.close()
        return [assessment for assessment in assessments if assessment is not None]

    def save(self, assessments: list[ScrapedAssessment]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        unique = {assessment.url: asdict(assessment) for assessment in assessments}
        self.output_path.write_text(json.dumps(list(unique.values()), indent=2, ensure_ascii=False), encoding="utf-8")

    def _discover_links(self, page: Page, max_pages: int) -> list[str]:
        seen: set[str] = set()
        queue = list(self.start_urls)
        assessment_links: set[str] = set()
        while queue and len(seen) < max_pages:
            url = queue.pop(0)
            if url in seen or not self._is_shl_url(url):
                continue
            seen.add(url)
            try:
                page.goto(url, wait_until="networkidle")
                self._accept_cookies(page)
                html = page.content()
            except Exception as exc:
                logger.warning("Playwright failed for %s, trying requests: %s", url, exc)
                html = requests.get(url, timeout=20).text
            soup = BeautifulSoup(html, "html.parser")
            for anchor in soup.select("a[href]"):
                href = anchor.get("href")
                if not href:
                    continue
                full_url = self._clean_url(urljoin(BASE_URL, href))
                if not self._is_shl_url(full_url):
                    continue
                if "/products/" in full_url and full_url not in seen and len(seen) + len(queue) < max_pages:
                    queue.append(full_url)
                if self._looks_like_assessment_url(full_url):
                    assessment_links.add(full_url)
        return sorted(assessment_links)

    def _scrape_assessment(self, page: Page, url: str) -> ScrapedAssessment | None:
        try:
            page.goto(url, wait_until="networkidle")
            html = page.content()
        except Exception as exc:
            logger.warning("Unable to load assessment %s: %s", url, exc)
            try:
                html = requests.get(url, timeout=20).text
            except requests.RequestException:
                return None

        soup = BeautifulSoup(html, "html.parser")
        name = self._first_text(soup, ["h1", "meta[property='og:title']", "title"])
        description = self._description(soup)
        if not name or len(name) < 2:
            return None
        page_text = soup.get_text(" ", strip=True)
        return ScrapedAssessment(
            name=self._clean_name(name),
            description=description,
            test_type=self._infer_test_type(page_text),
            category=self._infer_category(page_text, url),
            duration=self._extract_duration(page_text),
            languages=self._extract_languages(page_text),
            url=self._clean_url(url),
        )

    @staticmethod
    def _accept_cookies(page: Page) -> None:
        selectors = [
            "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
            "button:has-text('Allow all cookies')",
            "button:has-text('Accept All')",
            "button:has-text('Accept')",
        ]
        for selector in selectors:
            try:
                page.locator(selector).click(timeout=1500)
                return
            except Exception:
                continue

    @staticmethod
    def _first_text(soup: BeautifulSoup, selectors: list[str]) -> str:
        for selector in selectors:
            element = soup.select_one(selector)
            if element is None:
                continue
            value = element.get("content", "") if element.name == "meta" else element.get_text(" ", strip=True)
            if value:
                return str(value)
        return ""

    @staticmethod
    def _description(soup: BeautifulSoup) -> str:
        meta = soup.select_one("meta[name='description'], meta[property='og:description']")
        if meta and meta.get("content"):
            return str(meta["content"]).strip()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.select("p")]
        useful = [text for text in paragraphs if len(text) > 60]
        return useful[0] if useful else "SHL assessment catalog entry."

    @staticmethod
    def _infer_test_type(text: str) -> str:
        lowered = text.lower()
        candidates = [
            ("Personality", ["personality", "opq"]),
            ("Cognitive Ability", ["cognitive", "numerical", "verbal", "inductive", "deductive"]),
            ("Skills", ["skills", "coding", "technical", "simulation"]),
            ("Behavioral", ["behavioral", "situational judgement", "sjt"]),
            ("Motivation", ["motivation", "motivational"]),
            ("Leadership", ["leadership", "manager"]),
        ]
        for label, terms in candidates:
            if any(term in lowered for term in terms):
                return label
        return "Assessment"

    @staticmethod
    def _infer_category(text: str, url: str) -> str:
        lowered = f"{text} {url}".lower()
        for category in ["personality", "cognitive", "skills", "behavioral", "leadership", "graduate", "sales"]:
            if category in lowered:
                return category.title()
        return "SHL Assessment"

    @staticmethod
    def _extract_duration(text: str) -> str:
        match = re.search(r"\b(\d{1,3})\s*(minutes|mins|min)\b", text, flags=re.IGNORECASE)
        return f"{match.group(1)} minutes" if match else ""

    @staticmethod
    def _extract_languages(text: str) -> list[str]:
        match = re.search(r"languages?\s*[:\-]\s*([A-Za-z, ]{3,120})", text, flags=re.IGNORECASE)
        if not match:
            return []
        return [language.strip() for language in match.group(1).split(",") if language.strip()]

    @staticmethod
    def _clean_url(url: str) -> str:
        parsed = urlparse(url)
        return parsed._replace(fragment="", query="").geturl().rstrip("/")

    @staticmethod
    def _clean_name(name: str) -> str:
        return re.sub(r"\s+", " ", name.replace("| SHL", "")).strip()

    @staticmethod
    def _is_shl_url(url: str) -> bool:
        return urlparse(url).netloc.endswith("shl.com")

    @staticmethod
    def _looks_like_assessment_url(url: str) -> bool:
        lowered = url.lower()
        return "/products/" in lowered and any(term in lowered for term in ["assessment", "assessments", "product-catalog", "verify", "opq"])


def scrape_catalog(output_path: Path = OUTPUT_PATH) -> list[ScrapedAssessment]:
    scraper = SHLCatalogScraper(output_path=output_path)
    assessments = scraper.scrape()
    scraper.save(assessments)
    return assessments


if __name__ == "__main__":
    scraped = scrape_catalog()
    print(f"Saved {len(scraped)} SHL assessments to {OUTPUT_PATH}")
