import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from chat.routes import router as chat_router


app = FastAPI(title="Streamming APP")

app.include_router(chat_router)

app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8081, reload=True)
