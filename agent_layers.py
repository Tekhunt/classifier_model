import os
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
import asyncio
from decouple import config
from .retrieve_context.retrieve_helpers import ContextStoreDB

# Load environment variables
load_dotenv()
# os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', '')
DB_URL = config("DB_URL")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Adjust if you have a different valid model
    temperature=0,
    max_tokens=None,
    max_retries=2,
    api_key=config("GEMINI_API_KEY")
)

def intellify(source: Optional[str] = "context_store"):
    # Dynamic context retrieval
    async def retrieve():
        store_key = "context_store"  # Default to public context if no user
        db = ContextStoreDB(DB_URL)
        print(f"🔗 Connecting to DB: {DB_URL}, store_key: {store_key}")
        await db.connect()
        print(f"✅ Connected to {store_key} database.")
        context = await db.retrieve_context(table_name=store_key)
        return context["gemini_context"]

    # Tools (modified to accept user-specific context)
    def general_knowledge(question: str) -> str:
        """Answer questions based on user-specific knowledge base."""
        context = asyncio.run(retrieve())  # Fetch context dynamically
        print(f"🔍 Retrieved context for {source}")
        # print(f"📝 Context: {context}")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions based on this document:\n\n{document}"
                    "Please keep your answers concise/short (no more than 5 sentences) and relevant."),
            ("human", "{question}")
        ])
        chain = prompt | llm
        return chain.invoke({"document": context, "question": question}).content


    # Router Agent
    router_tools = [
        general_knowledge,
    ]


    # Swarm Setup
    checkpointer = InMemorySaver()
    store = InMemoryStore() 
    workflow = create_swarm(
        default_active_agent="Router"  
    )

    app = workflow.compile(
        checkpointer=checkpointer,
        store=store
    )

    # Test
    config = {"configurable": {"thread_id": "1"}}
    return app, config  


async def chat(question: str, source=None):
    app, config = intellify(source)
    result = app.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config,
    )
    res = result["messages"][-1].content
    return res

