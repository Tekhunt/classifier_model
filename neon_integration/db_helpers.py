import asyncpg
from datetime import datetime
from decouple import config
import os

DB_URL = os.getenv("DB_URL")

class LargeTextDB:
    """Handles database interactions for the large text storage."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = None

    async def connect(self):
        """Establish connection pool."""
        self.pool = await asyncpg.create_pool(self.db_url)

    async def close(self):
        """Close the database pool."""
        if self.pool:
            await self.pool.close()

    async def create_table(self, table_name: str):
        """Ensure the specified table exists with a single row for storage."""
        async with self.pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY CHECK (id = 1), 
                    content TEXT NOT NULL,  
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            await conn.execute(f"""
                INSERT INTO {table_name} (id, content) 
                VALUES (1, '') 
                ON CONFLICT (id) DO NOTHING;
            """)
            print(f"✅ Table '{table_name}' ensured with a single row.")

    async def update_text(self, new_content: str):
        """Append new content to the existing content in the database."""
        if not self.pool:
            raise RuntimeError("Database connection is not initialized. Call `await connect()` first.")

        async with self.pool.acquire() as conn:
            # Fetch the existing content
            existing_content = await conn.fetchval("SELECT content FROM context_store WHERE id = 1;")
            
            # Append the new content to the existing content
            separator = "\n\n"  # You can customize this separator
            updated_content = existing_content + separator + new_content

            # Update the database with the combined content
            await conn.execute("""
                UPDATE context_store 
                SET content = $1, updated_at = NOW()
                WHERE id = 1;
            """, updated_content)
            print(f"✅ Appended text at {datetime.now()}")


    async def read_text(self):
        """Retrieve the current stored text."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT content FROM context_store WHERE id = 1;")
            return row["content"] if row else None
        
    async def clear(self):
        """Clear the content of the table, resetting it to an empty string."""
        if not self.pool:
            raise RuntimeError("Database connection is not initialized. Call `await connect()` first.")

        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE context_store 
                SET content = '', updated_at = NOW()
                WHERE id = 1;
            """)
            print("✅ Cleared all data from the table.")

    async def rename_table(self, old_name: str, new_name: str):
        """Rename an existing table."""
        async with self.pool.acquire() as conn:
            await conn.execute(f"""
                ALTER TABLE {old_name} RENAME TO {new_name};
            """)
            print(f"✅ Table '{old_name}' renamed to '{new_name}'.")


async def table_exists():
    """Check if the context_store table exists."""
    db = LargeTextDB(DB_URL)
    await db.connect()
    
    async with db.pool.acquire() as conn:
        result = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'context_store'
            );
        """)
    
    await db.close()
    return result

async def check():
    exists = await table_exists()
    if exists:
        print("✅ Table 'context_store' exists.")
    else:
        print("❌ Table 'context_store' does NOT exist.")
    return exists


async def get_table_info():
    """Retrieve table name and configuration details."""
    db = LargeTextDB(DB_URL)
    await db.connect()
    
    async with db.pool.acquire() as conn:
        result = await conn.fetch("""
            SELECT 
                table_name, 
                table_schema, 
                table_type 
            FROM information_schema.tables 
            WHERE table_name = 'context_store';
        """)
    
    await db.close()
    return result

async def tinfo():
    table_info = await get_table_info()
    if table_info:
        print("✅ Table Details:")
        for row in table_info:
            print(f"Name: {row['table_name']}, Schema: {row['table_schema']}, Type: {row['table_type']}")
    else:
        print("❌ Table 'context_store' does NOT exist.")
    
