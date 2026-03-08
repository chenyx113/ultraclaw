"""
Tool manager for Ultra-Claw.

This module provides tool registration and execution management.
"""

from typing import Any, Dict, List, Optional, Type

from ultra_claw.services.tools.base import Tool, ToolResult, ToolParameter
from ultra_claw.utils.logger import get_logger

logger = get_logger(__name__)


class ToolManager:
    """
    Manager for agent tools.
    
    Provides registration, lookup, and execution of tools.
    """
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register default built-in tools."""
        # Register calculator tool
        self.register(CalculatorTool())
        
        # Register memory tool
        self.register(MemoryTool())
    
    def register(self, tool: Tool) -> None:
        """
        Register a tool.
        
        Args:
            tool: The tool to register
        """
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Tool name
            
        Returns:
            True if unregistered, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False
    
    def get(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            The tool, or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """
        List all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        Get schemas for all registered tools.
        
        Returns:
            List of tool schemas
        """
        return [tool.get_schema() for tool in self._tools.values()]
    
    async def execute(self, name: str, **kwargs) -> ToolResult:
        """
        Execute a tool.
        
        Args:
            name: Tool name
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                content="",
                error=f"Tool '{name}' not found"
            )
        
        try:
            result = await tool.execute(**kwargs)
            logger.debug(f"Executed tool {name}", extra={"result": result.success})
            return result
        except Exception as e:
            logger.error(f"Tool {name} execution failed: {e}")
            return ToolResult(
                success=False,
                content="",
                error=str(e)
            )


class CalculatorTool(Tool):
    """Tool for performing calculations."""
    
    name = "calculator"
    description = "Perform mathematical calculations"
    parameters = [
        ToolParameter(
            name="expression",
            description="The mathematical expression to evaluate",
            type="string",
            required=True
        ),
    ]
    
    async def execute(self, expression: str, **kwargs) -> ToolResult:
        """Execute a calculation."""
        try:
            # Safe evaluation - only allow basic math operations
            allowed_names = {
                "abs": abs,
                "max": max,
                "min": min,
                "round": round,
                "sum": sum,
                "pow": pow,
            }
            
            # Compile and evaluate
            code = compile(expression, "<string>", "eval")
            
            # Check for disallowed operations
            for name in code.co_names:
                if name not in allowed_names and not name.isdigit():
                    if name not in ("__builtins__",):
                        return ToolResult(
                            success=False,
                            content="",
                            error=f"Disallowed operation: {name}"
                        )
            
            result = eval(code, {"__builtins__": {}}, allowed_names)
            
            return ToolResult(
                success=True,
                content=str(result),
                metadata={"expression": expression, "result": result}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Calculation error: {str(e)}"
            )


class MemoryTool(Tool):
    """Tool for memory operations."""
    
    name = "memory"
    description = "Search and retrieve memories"
    parameters = [
        ToolParameter(
            name="operation",
            description="The operation to perform",
            type="string",
            required=True,
            enum=["search", "get", "store"]
        ),
        ToolParameter(
            name="query",
            description="The search query or memory content",
            type="string",
            required=True
        ),
        ToolParameter(
            name="category",
            description="Optional category filter",
            type="string",
            required=False
        ),
    ]
    
    async def execute(
        self,
        operation: str,
        query: str,
        category: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute a memory operation."""
        # This is a placeholder - in production, this would
        # interact with the memory service
        
        if operation == "search":
            return ToolResult(
                success=True,
                content=f"Searching memories for: {query}",
                metadata={"operation": operation, "query": query}
            )
        elif operation == "get":
            return ToolResult(
                success=True,
                content=f"Retrieving memory: {query}",
                metadata={"operation": operation, "id": query}
            )
        elif operation == "store":
            return ToolResult(
                success=True,
                content=f"Storing memory: {query}",
                metadata={"operation": operation, "content": query}
            )
        else:
            return ToolResult(
                success=False,
                content="",
                error=f"Unknown operation: {operation}"
            )
