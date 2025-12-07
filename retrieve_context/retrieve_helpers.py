import asyncpg
import os
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from dotenv import load_dotenv


DB_URL = config("DB_URL")

load_dotenv()
# os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', 'your-key-if-not-using-env')


class ContextStoreDB:
    """Handles interactions with the context_store table in NeonDB."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = None

    async def connect(self):
        """Establish the connection pool."""
        self.pool = await asyncpg.create_pool(self.db_url)

    async def close(self):
        """Close the database connection."""
        if self.pool:
            await self.pool.close()

    async def retrieve_context(self, table_name: str = "context_store"):
        """Fetch the context from the database and return it as a dictionary."""
        if not self.pool:
            raise RuntimeError("Database connection is not initialized. Call `await connect()` first.")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(f"SELECT content FROM {table_name} WHERE id = 1;")
            content = row["content"] if row else ""
            return {"gemini_context": content}
        
    async def update_context(self, new_data: str, table_name: str = "context_store"):
        """Update the context table with new appended data within a transaction."""
        if not self.pool:
            raise RuntimeError("Database connection is not initialized. Call `await connect()` first.")

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(f"SELECT content FROM {table_name} WHERE id = 1;")
                existing_data = row["content"] if row else ""
                merged_data = existing_data + "\n" + new_data.strip()

                await conn.execute(
                    f"UPDATE {table_name} SET content = $1 WHERE id = 1;",
                    merged_data
                )


def create_prompt(document: str, question) -> ChatPromptTemplate:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=config("GEMINI_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful assistant that answers questions based on the following document:\n\n"
                    "{document}\n\n"
                    "Please keep your answers concise (no more than 5 sentences) and ensure your answers "
                    "are relevant to the user's question. If you don't know the answer, say 'I don't know'."
                ),
            ),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm
    result = chain.invoke(
        {
            "document": document,
            "question": question,
        }
    )
    print(f"llm used: {llm}")
    return result.content

async def retrieve(table_name: str = "context_store") -> dict:
    """Retrieve the Gemini context from NeonDB and return it as a dictionary."""
    db = ContextStoreDB(DB_URL)
    await db.connect()
    context = await db.retrieve_context(table_name)
    await db.close()
    return context

async def generate(question: str, table_name: str = "context_store") -> str:
    """Generate a response based on the retrieved context and the user's question."""
    context = await retrieve(table_name)  # Retrieve the context from the database
    document = context.get("gemini_context", "")  # Extract the content safely
    res = create_prompt(document, question)  # Call the synchronous prompt function
    return res

