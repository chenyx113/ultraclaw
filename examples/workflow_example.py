"""
Workflow example for Ultra-Claw.

This example demonstrates how to create and execute workflows.
"""

import asyncio

from ultra_claw import UltraAgent
from ultra_claw.core.models import AgentConfig, WorkflowStep


async def main():
    """Run the workflow example."""
    config = AgentConfig(
        name="Workflow Agent",
        llm={"provider": "mock"}
    )
    
    agent = UltraAgent(config=config)
    await agent.initialize()
    
    print("Creating a workflow...\n")
    
    # Define workflow steps
    steps = [
        WorkflowStep(
            name="log_start",
            action="log",
            parameters={
                "message": "Starting workflow execution",
                "level": "info"
            }
        ),
        WorkflowStep(
            name="set_variables",
            action="set_variable",
            parameters={
                "name": "greeting",
                "value": "Hello, World!"
            }
        ),
        WorkflowStep(
            name="log_greeting",
            action="log",
            parameters={
                "message": "Greeting set: ${greeting}",
                "level": "info"
            },
            dependencies=["set_variables"]
        ),
        WorkflowStep(
            name="sleep_step",
            action="sleep",
            parameters={
                "duration": 0.5
            },
            dependencies=["log_start"]
        ),
        WorkflowStep(
            name="log_end",
            action="log",
            parameters={
                "message": "Workflow completed successfully!",
                "level": "info"
            },
            dependencies=["log_greeting", "sleep_step"]
        ),
    ]
    
    print(f"Executing workflow with {len(steps)} steps...\n")
    
    # Execute the workflow
    state = await agent.execute_workflow(
        steps=steps,
        context={"start_time": "now"}
    )
    
    print(f"\nWorkflow completed!")
    print(f"  State ID: {state.id}")
    print(f"  Status: {state.status}")
    print(f"  Completed steps: {len(state.completed_steps)}/{len(steps)}")
    print(f"  Failed steps: {len(state.failed_steps)}")
    print(f"  Results: {state.results}")
    
    # Cleanup
    await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
