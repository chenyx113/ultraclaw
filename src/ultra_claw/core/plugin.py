"""
Plugin system for Ultra-Claw.

This module provides a plugin architecture for extending Ultra-Claw's functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type

from ultra_claw.core.models import MemoryItem
from ultra_claw.utils.logger import get_logger

logger = get_logger(__name__)


class Plugin(ABC):
    """
    Abstract base class for plugins.
    
    All plugins must implement this interface.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the plugin.
        
        Args:
            config: Plugin configuration
            
        Returns:
            True if initialization succeeded
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Shutdown the plugin.
        
        Returns:
            True if shutdown succeeded
        """
        pass


class MemoryPlugin(Plugin):
    """Base class for memory backend plugins."""
    
    @abstractmethod
    async def store(self, item: MemoryItem) -> str:
        """Store a memory item."""
        pass
    
    @abstractmethod
    async def retrieve(self, query: str, filters: Dict[str, Any]) -> List[MemoryItem]:
        """Retrieve memories."""
        pass
    
    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """Delete a memory item."""
        pass


class PluginManager:
    """
    Manager for plugins.
    
    Handles plugin registration, lifecycle, and hook system.
    """
    
    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._hooks: Dict[str, List[Callable]] = {}
    
    async def load_plugin(
        self,
        plugin_class: Type[Plugin],
        config: Dict[str, Any]
    ) -> bool:
        """
        Load and initialize a plugin.
        
        Args:
            plugin_class: Plugin class to load
            config: Plugin configuration
            
        Returns:
            True if plugin loaded successfully
        """
        plugin = plugin_class()
        
        try:
            if await plugin.initialize(config):
                self._plugins[plugin.name] = plugin
                self._register_hooks(plugin)
                logger.info(f"Loaded plugin: {plugin.name} v{plugin.version}")
                return True
            else:
                logger.error(f"Failed to initialize plugin: {plugin.name}")
                return False
        except Exception as e:
            logger.error(f"Error loading plugin {plugin.name}: {e}")
            return False
    
    async def unload_plugin(self, name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            True if plugin was unloaded
        """
        plugin = self._plugins.get(name)
        if not plugin:
            return False
        
        try:
            await plugin.shutdown()
            del self._plugins[name]
            logger.info(f"Unloaded plugin: {name}")
            return True
        except Exception as e:
            logger.error(f"Error unloading plugin {name}: {e}")
            return False
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Get a loaded plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance, or None if not loaded
        """
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """
        List all loaded plugins.
        
        Returns:
            List of plugin names
        """
        return list(self._plugins.keys())
    
    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """
        Register a hook callback.
        
        Args:
            hook_name: Name of the hook
            callback: Function to call when hook is triggered
        """
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)
        logger.debug(f"Registered hook: {hook_name}")
    
    async def trigger_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Trigger a hook.
        
        Args:
            hook_name: Name of the hook to trigger
            *args: Positional arguments to pass to callbacks
            **kwargs: Keyword arguments to pass to callbacks
            
        Returns:
            List of callback results
        """
        results = []
        callbacks = self._hooks.get(hook_name, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(*args, **kwargs)
                else:
                    result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook {hook_name} callback failed: {e}")
        
        return results
    
    def _register_hooks(self, plugin: Plugin) -> None:
        """Register hooks from a plugin."""
        # This can be extended to auto-discover hooks from plugins
        pass


# Import asyncio here for the trigger_hook method
import asyncio
