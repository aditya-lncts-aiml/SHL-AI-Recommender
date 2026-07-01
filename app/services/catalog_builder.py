from __future__ import annotations

from app.services.scraper import scrape_catalog


def main() -> None:
    assessments = scrape_catalog()
    print(f"Catalog built with {len(assessments)} SHL assessments")


if __name__ == "__main__":
    main()
