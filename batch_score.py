"""Batch scoring script.

Reads a CSV with `subject` and `body` columns, routes each ticket using
`route_ticket` and writes `data/routed_tickets.csv` with added columns.
"""
import sys
import pandas as pd
from tqdm import tqdm
import json

from routing import route_ticket


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/new_tickets.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Input file not found: {path}")
        return

    # Ensure expected columns
    if not all(c in df.columns for c in ["subject", "body"]):
        print("Input CSV must contain 'subject' and 'body' columns")
        return

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Routing tickets"):
        subj = row.get("subject", "")
        body = row.get("body", "")
        res = route_ticket(subj, body)
        results.append(res)

    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    # Normalize assigned_departments to JSON string for CSV
    out["assigned_departments"] = out["assigned_departments"].apply(lambda x: json.dumps(x))

    out_path = "data/routed_tickets.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved routed tickets to {out_path}")

    # Print summary counts per department
    dept_counts = {}
    for lst in out["assigned_departments"]:
        for d in json.loads(lst):
            dept_counts[d] = dept_counts.get(d, 0) + 1

    print("Routing summary:")
    for dept, count in sorted(dept_counts.items(), key=lambda x: -x[1]):
        print(f"- {dept}: {count}")


if __name__ == "__main__":
    main()
