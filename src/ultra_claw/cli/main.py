"""
Command-line interface for Ultra-Claw.

This module provides CLI commands for interacting with Ultra-Claw.
"""

import asyncio
import json
import sys
from pathlib import Path

import click

from ultra_claw.core.agent import UltraAgent
from ultra_claw.core.models import AgentConfig, Message
from ultra_claw.utils.config import load_config, save_config
from ultra_claw.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


@click.group()
@click.option("--config", "-c", help="Path to configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, config, verbose):
    """Ultra-Claw: Next-generation intelligent agent framework."""
    ctx.ensure_object(dict)
    
    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(level=log_level)
    
    # Load configuration
    ctx.obj["config_path"] = config
    ctx.obj["config"] = load_config(config)


@cli.command()
@click.pass_context
def serve(ctx):
    """Start the API server."""
    import uvicorn
    from ultra_claw.api.main import app
    
    click.echo("Starting Ultra-Claw API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


@cli.command()
@click.argument("message")
@click.option("--user-id", "-u", default="cli-user", help="User ID")
@click.option("--session-id", "-s", help="Session ID")
@click.option("--no-memory", is_flag=True, help="Disable memory retrieval")
@click.pass_context
def chat(ctx, message, user_id, session_id, no_memory):
    """Chat with the agent."""
    config = ctx.obj["config"]
    
    async def _chat():
        agent = UltraAgent(config=config)
        await agent.initialize()
        
        messages = [Message(role="user", content=message)]
        
        click.echo(f"\nYou: {message}\n")
        click.echo("Agent: ", nl=False)
        
        response_text = ""
        async for response in agent.chat(
            messages=messages,
            session_id=session_id,
            user_id=user_id,
            use_memory=not no_memory
        ):
            click.echo(response.content, nl=False)
            response_text += response.content
        
        click.echo("\n")
        
        if response.memories_used:
            click.echo(f"\n[Used {len(response.memories_used)} memories]")
        
        await agent.shutdown()
    
    asyncio.run(_chat())


@cli.command()
@click.argument("content")
@click.option("--user-id", "-u", required=True, help="User ID")
@click.option("--category", "-c", multiple=True, help="Categories")
@click.option("--tag", "-t", multiple=True, help="Tags")
@click.option("--importance", "-i", type=float, default=0.5, help="Importance score")
@click.pass_context
def memorize(ctx, content, user_id, category, tag, importance):
    """Store a memory."""
    config = ctx.obj["config"]
    
    async def _memorize():
        agent = UltraAgent(config=config)
        await agent.initialize()
        
        item = await agent.memorize(
            content=content,
            user_id=user_id,
            categories=list(category),
            tags=list(tag),
            importance=importance
        )
        
        click.echo(f"Stored memory: {item.id}")
        
        await agent.shutdown()
    
    asyncio.run(_memorize())


@cli.command()
@click.argument("query")
@click.option("--user-id", "-u", help="User ID")
@click.option("--category", "-c", multiple=True, help="Categories")
@click.option("--top-k", "-k", type=int, default=10, help="Number of results")
@click.pass_context
def retrieve(ctx, query, user_id, category, top_k):
    """Retrieve memories."""
    config = ctx.obj["config"]
    
    async def _retrieve():
        agent = UltraAgent(config=config)
        await agent.initialize()
        
        memories = await agent.retrieve(
            query=query,
            user_id=user_id,
            categories=list(category) if category else None,
            top_k=top_k
        )
        
        if not memories:
            click.echo("No memories found.")
        else:
            click.echo(f"\nFound {len(memories)} memories:\n")
            for i, memory in enumerate(memories, 1):
                click.echo(f"{i}. [{memory.id}]")
                click.echo(f"   Content: {memory.content[:100]}...")
                click.echo(f"   Categories: {', '.join(memory.categories) or 'None'}")
                click.echo(f"   Importance: {memory.importance_score}")
                click.echo()
        
        await agent.shutdown()
    
    asyncio.run(_retrieve())


@cli.command()
@click.argument("memory_id")
@click.pass_context
def forget(ctx, memory_id):
    """Delete a memory."""
    config = ctx.obj["config"]
    
    async def _forget():
        agent = UltraAgent(config=config)
        await agent.initialize()
        
        success = await agent.forget(memory_id)
        
        if success:
            click.echo(f"Deleted memory: {memory_id}")
        else:
            click.echo(f"Memory not found: {memory_id}")
        
        await agent.shutdown()
    
    asyncio.run(_forget())


@cli.command()
@click.option("--output", "-o", help="Output file path")
@click.pass_context
def config(ctx, output):
    """Show or save configuration."""
    config = ctx.obj["config"]
    config_dict = config.model_dump()
    
    if output:
        save_config(config, output)
        click.echo(f"Configuration saved to {output}")
    else:
        click.echo(json.dumps(config_dict, indent=2, default=str))


@cli.command()
@click.pass_context
def stats(ctx):
    """Show agent statistics."""
    config = ctx.obj["config"]
    
    async def _stats():
        agent = UltraAgent(config=config)
        await agent.initialize()
        
        stats = await agent.get_stats()
        click.echo(json.dumps(stats, indent=2, default=str))
        
        await agent.shutdown()
    
    asyncio.run(_stats())


@cli.command()
def version():
    """Show version information."""
    from ultra_claw import __version__
    click.echo(f"Ultra-Claw version {__version__}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
