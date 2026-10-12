import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from app_backend import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
