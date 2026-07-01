from app.services.retrieval import Retriever

r = Retriever()

print(f"Total assessments: {len(r.catalog)}")

found = False

for a in r.catalog:
    if (
        "personality" in a.search_text().lower()
        or "opq" in a.name.lower()
    ):
        found = True
        print("=" * 60)
        print("Name:", a.name)
        print("Test Type:", a.test_type)
        print("Search Text:", a.search_text())

if not found:
    print("\nNo personality/OPQ assessments found in the catalog.")