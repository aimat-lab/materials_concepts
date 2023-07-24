from fastapi import FastAPI


app = FastAPI()


@app.get("/search")
def search(query: str, semantic_search: bool = False):
    return {"message": f"You searched for {query}"}
