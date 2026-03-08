# Ultra-Claw

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ultra-Claw is a next-generation intelligent agent framework with long-term memory and multimodal capabilities. It integrates the best of OpenClaw's architecture, MemU's memory plugin capabilities, and Nanobot's Python development philosophy.

## Features

- **Long-term Memory**: Persistent cross-session memory with vector-based semantic search
- **Multimodal Support**: Handle text, images, audio, and video memories
- **Hybrid Retrieval**: Combine vector, keyword, and temporal search
- **Workflow Engine**: Execute complex multi-step tasks with dependencies
- **Session Management**: Track conversations with context preservation
- **Plugin System**: Extensible architecture for custom functionality
- **RESTful API**: Full-featured API for integration
- **CLI Tools**: Command-line interface for easy interaction
- **Multiple LLM Providers**: Support for OpenAI, Anthropic, and more

## Installation

### From Source

```bash
git clone https://github.com/chenyx113/ultraclaw.git
cd ultra-claw
pip install -e .
```

### With Optional Dependencies

```bash
# Install with all optional features
pip install -e ".[all]"

# Install development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
import asyncio
from ultra_claw import UltraAgent
from ultra_claw.core.models import AgentConfig, Message

async def main():
    # Create an agent
    config = AgentConfig(
        name="My Assistant",
        llm={"provider": "openai", "model": "gpt-4"}
    )
    agent = UltraAgent(config=config)
    await agent.initialize()
    
    # Store a memory
    await agent.memorize(
        content="I love hiking in the mountains",
        user_id="user-123",
        categories=["preferences"]
    )
    
    # Chat with memory
    messages = [Message(role="user", content="What do I enjoy doing?")]
    async for response in agent.chat(messages, user_id="user-123"):
        print(response.content, end="")
    
    await agent.shutdown()

asyncio.run(main())
```

### Using the CLI

```bash
# Start the API server
ultra-claw serve

# Chat with the agent
ultra-claw chat "Hello, how are you?" --user-id myuser

# Store a memory
ultra-claw memorize "Important information" --user-id myuser --category knowledge

# Retrieve memories
ultra-claw retrieve "important" --user-id myuser
```

### Using the API

```bash
# Start the server
ultra-claw serve

# Create an agent
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"name": "My Agent", "config": {"llm": {"provider": "openai"}}}'

# Chat
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "user_id": "user-123"
  }'

# Store memory
curl -X POST http://localhost:8000/api/v1/memory \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Remember this",
    "user_id": "user-123",
    "categories": ["knowledge"]
  }'
```

## Configuration

Create a configuration file at `config/ultra-claw.yaml`:

```yaml
agent:
  name: "Ultra-Claw Agent"

memory:
  backend: "sqlite"
  database_url: "sqlite:///memory.db"
  max_context_memories: 10

llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7

workflow:
  max_steps: 50
  timeout: 300
```

Set environment variables:

```bash
export OPENAI_API_KEY="your-key"
export ULTRACLAW_CONFIG_PATH="/path/to/config.yaml"
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t ultra-claw .
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY ultra-claw
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │  CLI 工具   │  │  Web 界面   │  │  API 服务   │  │  插件   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                        Service Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │  会话管理   │  │  记忆服务   │  │  工具管理   │  │  安全    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                        Core Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │  代理引擎   │  │  工作流引擎 │  │  事件总线   │  │  配置    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ultra_claw

# Run specific test file
pytest tests/unit/test_memory.py

# Run integration tests
pytest tests/integration/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by [OpenClaw](https://github.com/open-claw/openclaw)
- Memory system inspired by [MemU](https://github.com/memU-project/memU)
- Architecture influenced by [Nanobot](https://github.com/nanobot-framework/nanobot)

## Support

For support, please open an issue on GitHub or contact the maintainers.
