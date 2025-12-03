import asyncio
import os
from db_helpers import LargeTextDB, DB_URL
from decouple import config


DB_URL = os.getenv("DB_URL")
DB_URL = config("DB_URL")

# TABLE_NAME = "dnwine_store"
TABLE_NAME = "context_store"  # Change this to the desired table name


async def setup_database():
    """Creates the content table if it does not exist."""
    print(f"🔗 Connecting to DB: {DB_URL}") 
    db = LargeTextDB(DB_URL)
    
    try:
        await db.connect()
        await db.create_table(TABLE_NAME) 
        print(f"✅ Table '{TABLE_NAME}' setup completed successfully.")
    except Exception as e:
        print(f"❌ Error during table creation: {e}")
    finally:
        await db.close()

def run():
    asyncio.run(setup_database())

if __name__ == "__main__":
    run()

