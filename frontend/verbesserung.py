import pandas as pd
import json

df = pd.read_csv("books.csv")

# Keep only necessary columns
df = df[["book_id", "title", "authors", "average_rating"]]
df["book_id"] = df["book_id"].astype(str)

metadata_dict = {
    row["book_id"]: {
        "title": row["title"],
        "author": row["authors"],
        "avg_rating": float(row["average_rating"])
    }
    for _, row in df.iterrows()
}

with open("book_metadata.json", "w") as f:
    json.dump(metadata_dict, f, indent=2)
