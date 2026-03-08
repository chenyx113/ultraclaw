"""
Base tool interface for Ultra-Claw.

This module defines the abstract interface for tools that agents can use.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    
    name: str
    description: str
    type: str = "string"
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None


class ToolResult(BaseModel):
    """Result of a tool execution."""
    
    success: bool
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class Tool(ABC):
    """
    Abstract base class for tools.
    
    Tools are actions that an agent can perform, such as
    searching the web, calculating values, or calling APIs.
    """
    
    name: str = ""
    description: str = ""
    parameters: List[ToolParameter] = []
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
