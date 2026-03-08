"""
Basic usage example for Ultra-Claw.

This example demonstrates the core features of Ultra-Claw:
- Creating an agent
- Storing and retrieving memories
- Having a conversation
"""

import asyncio

from ultra_claw import UltraAgent
from ultra_claw.core.models import AgentConfig, Message


async def main():
    """Run the basic usage example."""
    # Create an agent with default configuration
    config = AgentConfig(
        name="Personal Assistant",
        llm={
            "provider": "mock",  # Use mock for demo
            "model": "gpt-4",
            "responses": [
                "Hello! I'm your personal assistant. How can I help you today?",
                "That's interesting! Let me remember that.",
                "Based on what you've told me before, I think you'd enjoy that!"
            ]
        }
    )
    
    agent = UltraAgent(config=config)
    await agent.initialize()
    
    print(f"Created agent: {agent.config.name}\n")
    
    # Store some memories
    print("Storing memories...")
    
    await agent.memorize(
        content="User's name is Alice and they love hiking",
        user_id="alice",
        categories=["personal", "preferences"],
        importance=0.9
    )
    
    await agent.memorize(
        content="Alice's favorite hiking spot is Mount Rainier",
        user_id="alice",
        categories=["personal", "preferences"],
        importance=0.8
    )
    
    await agent.memorize(
        content="Python is a versatile programming language",
        user_id="alice",
        categories=["knowledge", "programming"],
        importance=0.6
    )
    
    print("Memories stored!\n")
    
    # Retrieve memories
    print("Retrieving memories about hiking...")
    memories = await agent.retrieve(
        query="hiking",
        user_id="alice",
        top_k=5
    )
    
    for i, memory in enumerate(memories, 1):
        print(f"{i}. {memory.content}")
    
    print()
    
    # Have a conversation
    print("Having a conversation...\n")
    
    messages = [
        Message(role="user", content="Hi, do you remember my name?")
    ]
    
    print(f"User: {messages[0].content}")
    print("Agent: ", end="", flush=True)
    
    async for response in agent.chat(
        messages=messages,
        user_id="alice",
        use_memory=True
    ):
        print(response.content, end="", flush=True)
    
    print("\n")
    
    # Show agent statistics
    print("Agent Statistics:")
    stats = await agent.get_stats()
    print(f"  Total memories: {stats['memory']['total_items']}")
    print(f"  Total sessions: {stats['sessions']['total_sessions']}")
    print(f"  Active sessions: {stats['sessions']['active_sessions']}")
    
    # Cleanup
    await agent.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
