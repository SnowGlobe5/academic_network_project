import pandas as pd

count_total = 0
count_duplicate = 0
authors_name = {}

df = pd.read_csv("20230425233353/mapping/authors.csv", quotechar='"')
for i, line in enumerate(df['name']):
    if authors_name.get(line):
        count_duplicate += 1
        print(f"{line} - {i}")
    else:
        authors_name[line] = True
    count_total += 1

print(f"total:{count_total}, duplicate: {count_duplicate}, ratio:{count_duplicate/count_total}")
