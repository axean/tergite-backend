import random

n_nodes = 10000

contents = ""
newline = "\n"

OUTPUT_FILE = "big_graph.json"

if __name__ == "__main__":
    print(f"Creating output file {OUTPUT_FILE} ...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        contents += "{"
        first = True
        for i in range(n_nodes):
            if not first:
                contents += ",\n"
            else:
                first = False

            contents += f"""
  "node_{i}":{{
    "dependencies": [
      {'"node_' + str(i+1) + '"' if i+1 < n_nodes else ''}{',' + newline + '      "node_' + str(i+2) + '"' if i+2 < n_nodes else ''}{',' + newline + '      "node_' + str(i+3) + '"' if i+3 < n_nodes else ''}
    ],
    "function": "run_fun{i}",
    "params": [
      {{
        "name": "p{i}",
        "unit": "s",
        "threshold_upper": 22e-6,
        "threshold_lower": 20e-6,
        "timeout": {int(random.random() * 50 + 10)}
      }}
    ]
  }}
"""
        contents += "}"
        file.write(contents)

    print("Done")
