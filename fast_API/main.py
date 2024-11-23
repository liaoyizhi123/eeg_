from fastapi import FastAPI
import uvicorn


app = FastAPI()

@app.get("/", tags=["home"])
async def home():
    return {
        "code": 2122,
        "message": "Hello World!"
    }

# if __name__ == '__main__':
#     uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True,  log_level="info")

#