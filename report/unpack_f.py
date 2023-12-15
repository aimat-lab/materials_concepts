import pickle

key = "pascal_friederich.txt"

with open("authors.pickle", "rb") as f:
    data = pickle.load(f)["pascal_friederich.txt"]

with open("friederich.pickle", "wb") as f:
    pickle.dump(data, f)
