import asyncio
import asyncpg
import os
from decouple import config
from db_helpers import DB_URL

table_name = "context_store" 
DB_URL = os.getenv("DB_URL", DB_URL)
DOCUMENT_PATH = "nb.txt"
DB_URL = config("DB_URL")

async def write_document_to_table(table_name: str = "context_store"):
    """Reads document and writes/updates it in the database."""
    try:
        # Read document.txt
        with open("nb.txt", "r", encoding="utf-8") as file:
            content = file.read()

        # Connect to the database
        conn = await asyncpg.connect(DB_URL)

        # Check if a row exists
        existing = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name} WHERE id = 1;") #context_store WHERE id = 1;")

        if existing > 0:
            # Delete the existing row
            await conn.execute(f"DELETE FROM {table_name} WHERE id = 1;")
            print("✅ Deleted existing row.")

        # Insert new row
        await conn.execute(f"INSERT INTO {table_name} (id, content) VALUES (1, $1);", content)
        print("✅ Inserted new document into database.")

        await conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(write_document_to_table(table_name))

