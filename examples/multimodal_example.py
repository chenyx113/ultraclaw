"""
Multimodal memory example for Ultra-Claw.

This example demonstrates storing and retrieving different types of memories.
"""

import asyncio

from ultra_claw import UltraAgent
from ultra_claw.core.models import AgentConfig, ModalityType


async def main():
    """Run the multimodal example."""
    config = AgentConfig(
        name="Multimodal Agent",
        llm={"provider": "mock"}
    )
    
    agent = UltraAgent(config=config)
    await agent.initialize()
    
    print("Storing multimodal memories...\n")
    
    # Store text memory
    text_memory = await agent.memorize(
        content="Meeting notes: Discussed Q4 goals and roadmap",
        user_id="user-1",
        modality=ModalityType.TEXT,
        categories=["work", "meetings"],
        importance=0.8
    )
    print(f"Stored text memory: {text_memory.id}")
    
    # Store image memory (simulated)
    image_memory = await agent.memorize(
        content="[Image] Screenshot of dashboard showing metrics",
        user_id="user-1",
        modality=ModalityType.IMAGE,
        categories=["work", "dashboards"],
        tags=["screenshot", "metrics"],
        importance=0.7
    )
    print(f"Stored image memory: {image_memory.id}")
    
    # Store audio memory (simulated)
    audio_memory = await agent.memorize(
        content="[Audio] Voice memo: Remember to call John about the project",
        user_id="user-1",
        modality=ModalityType.AUDIO,
        categories=["personal", "reminders"],
        importance=0.9
    )
    print(f"Stored audio memory: {audio_memory.id}")
    
    # Store video memory (simulated)
    video_memory = await agent.memorize(
        content="[Video] Recording of team standup meeting",
        user_id="user-1",
        modality=ModalityType.VIDEO,
        categories=["work", "meetings"],
        importance=0.6
    )
    print(f"Stored video memory: {video_memory.id}")
    
    print("\nRetrieving memories by modality...\n")
    
    # Get memory statistics
    stats = await agent.memory.get_stats(user_id="user-1")
    print("Memory Statistics:")
    for modality, count in stats["by_modality"].items():
        print(f"  {modality}: {count}")
    
    print("\nSearching for meeting-related memories...")
    meeting_memories = await agent.retrieve(
        query="meeting",
        user_id="user-1",
        top_k=10
    )
    
    for i, memory in enumerate(meeting_memories, 1):
        print(f"{i}. [{memory.modality.value}] {memory.content[:50]}...")
    
    # Cleanup
    await agent.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
