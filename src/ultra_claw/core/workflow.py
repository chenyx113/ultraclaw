"""
Workflow Engine for Ultra-Claw.

This module provides workflow management capabilities, allowing
complex multi-step tasks to be defined and executed with support
for dependencies, conditions, and error handling.
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type
from uuid import uuid4

from ultra_claw.core.models import WorkflowStep, WorkflowState, WorkflowConfig
from ultra_claw.utils.logger import get_logger

logger = get_logger(__name__)


class WorkflowAction:
    """Base class for workflow actions."""
    
    name: str = ""
    description: str = ""
    
    async def execute(self, context: Dict[str, Any], **parameters) -> Dict[str, Any]:
        """Execute the action."""
        raise NotImplementedError


class WorkflowRegistry:
    """Registry for workflow actions."""
    
    def __init__(self):
        self._actions: Dict[str, Type[WorkflowAction]] = {}
    
    def register(self, action_class: Type[WorkflowAction]) -> None:
        """Register an action class."""
        self._actions[action_class.name] = action_class
    
    def get(self, name: str) -> Optional[Type[WorkflowAction]]:
        """Get an action class by name."""
        return self._actions.get(name)
    
    def list_actions(self) -> List[str]:
        """List all registered action names."""
        return list(self._actions.keys())


# Global registry
workflow_registry = WorkflowRegistry()


def register_action(action_class: Type[WorkflowAction]) -> Type[WorkflowAction]:
    """Decorator to register a workflow action."""
    workflow_registry.register(action_class)
    return action_class


@register_action
class LogAction(WorkflowAction):
    """Action to log a message."""
    
    name = "log"
    description = "Log a message to the console"
    
    async def execute(self, context: Dict[str, Any], **parameters) -> Dict[str, Any]:
        message = parameters.get("message", "")
        level = parameters.get("level", "info")
        
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug":
            logger.debug(message)
        
        return {"success": True, "message": message}


@register_action
class SleepAction(WorkflowAction):
    """Action to sleep for a specified duration."""
    
    name = "sleep"
    description = "Sleep for a specified duration"
    
    async def execute(self, context: Dict[str, Any], **parameters) -> Dict[str, Any]:
        duration = parameters.get("duration", 1.0)
        await asyncio.sleep(duration)
        return {"success": True, "duration": duration}


@register_action
class SetVariableAction(WorkflowAction):
    """Action to set a variable in the context."""
    
    name = "set_variable"
    description = "Set a variable in the workflow context"
    
    async def execute(self, context: Dict[str, Any], **parameters) -> Dict[str, Any]:
        name = parameters.get("name")
        value = parameters.get("value")
        
        if name is None:
            return {"success": False, "error": "Variable name is required"}
        
        context[name] = value
        return {"success": True, "name": name, "value": value}


@register_action
class ConditionAction(WorkflowAction):
    """Action to evaluate a condition."""
    
    name = "condition"
    description = "Evaluate a condition and return result"
    
    async def execute(self, context: Dict[str, Any], **parameters) -> Dict[str, Any]:
        condition = parameters.get("condition", "")
        
        # Simple condition evaluation
        try:
            # Replace context variables
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool)):
                    condition = condition.replace(f"${{{key}}}", str(value))
            
            # Evaluate (in production, use a safer evaluator)
            result = eval(condition, {"__builtins__": {}}, {})
            return {"success": True, "result": bool(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}


class WorkflowEngine:
    """
    Workflow engine for executing multi-step workflows.
    
    Provides support for:
    - Step dependencies
    - Conditional execution
    - Error handling and retries
    - Parallel execution where possible
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self.registry = workflow_registry
        self._workflows: Dict[str, List[WorkflowStep]] = {}
        self._states: Dict[str, WorkflowState] = {}
        self._lock = asyncio.Lock()
    
    def register_workflow(self, workflow_id: str, steps: List[WorkflowStep]) -> None:
        """
        Register a workflow definition.
        
        Args:
            workflow_id: Unique identifier for the workflow
            steps: List of workflow steps
        """
        self._workflows[workflow_id] = steps
        logger.info(f"Registered workflow {workflow_id} with {len(steps)} steps")
    
    async def execute(
        self,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None,
        state_id: Optional[str] = None
    ) -> WorkflowState:
        """
        Execute a workflow.
        
        Args:
            workflow_id: The workflow to execute
            context: Initial context variables
            state_id: Optional state ID for resuming
            
        Returns:
            The final workflow state
        """
        steps = self._workflows.get(workflow_id)
        if not steps:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Create or resume state
        if state_id and state_id in self._states:
            state = self._states[state_id]
            state.status = "running"
        else:
            state = WorkflowState(
                workflow_id=workflow_id,
                context=context or {}
            )
        
        self._states[state.id] = state
        
        try:
            # Build dependency graph
            step_map = {step.id: step for step in steps}
            pending = set(step.id for step in steps if step.id not in state.completed_steps)
            
            start_time = datetime.utcnow()
            
            while pending:
                # Check timeout
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                if elapsed > self.config.timeout:
                    state.status = "failed"
                    state.error_message = "Workflow timeout"
                    break
                
                # Find ready steps (dependencies satisfied)
                ready = []
                for step_id in pending:
                    step = step_map[step_id]
                    deps_satisfied = all(
                        dep in state.completed_steps for dep in step.dependencies
                    )
                    if deps_satisfied:
                        ready.append(step)
                
                if not ready:
                    if pending:
                        # Deadlock detected
                        state.status = "failed"
                        state.error_message = "Dependency deadlock detected"
                    break
                
                # Execute ready steps
                tasks = [self._execute_step(step, state) for step in ready]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for step, result in zip(ready, results):
                    pending.discard(step.id)
                    
                    if isinstance(result, Exception):
                        state.failed_steps.append(step.id)
                        if step.on_error == "continue":
                            continue
                        else:
                            state.status = "failed"
                            state.error_message = str(result)
                            break
                    else:
                        state.completed_steps.append(step.id)
                        state.results[step.name] = result
                
                if state.status == "failed":
                    break
            
            # Update final state
            if state.status != "failed":
                if pending:
                    state.status = "failed"
                    state.error_message = f"Incomplete workflow, {len(pending)} steps pending"
                else:
                    state.status = "completed"
            
            state.end_time = datetime.utcnow()
            
        except Exception as e:
            state.status = "failed"
            state.error_message = str(e)
            state.end_time = datetime.utcnow()
            logger.error(f"Workflow {workflow_id} failed: {e}")
        
        return state
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        state: WorkflowState
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        state.current_step = step.id
        
        # Check condition
        if step.condition:
            condition_result = await self._evaluate_condition(step.condition, state.context)
            if not condition_result:
                logger.debug(f"Skipping step {step.name} due to condition")
                return {"skipped": True, "reason": "condition_not_met"}
        
        # Get action
        action_class = self.registry.get(step.action)
        if not action_class:
            raise ValueError(f"Unknown action: {step.action}")
        
        action = action_class()
        
        # Execute with retries
        last_error = None
        for attempt in range(self.config.retry_attempts + 1):
            try:
                # Apply timeout if specified
                timeout = step.timeout or self.config.timeout
                
                result = await asyncio.wait_for(
                    action.execute(state.context, **step.parameters),
                    timeout=timeout
                )
                
                logger.debug(f"Executed step {step.name}", extra={
                    "step": step.name,
                    "action": step.action,
                    "attempt": attempt + 1
                })
                
                return result
                
            except asyncio.TimeoutError:
                last_error = f"Step {step.name} timed out"
                logger.warning(f"Step {step.name} timed out (attempt {attempt + 1})")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Step {step.name} failed (attempt {attempt + 1}): {e}")
            
            if attempt < self.config.retry_attempts:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        raise Exception(f"Step {step.name} failed after {self.config.retry_attempts + 1} attempts: {last_error}")
    
    async def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition string."""
        try:
            # Replace context variables
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool)):
                    condition = condition.replace(f"${{{key}}}", str(value))
            
            # Simple evaluation (in production, use a safer evaluator)
            result = eval(condition, {"__builtins__": {}}, {})
            return bool(result)
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {e}")
            return False
    
    def get_state(self, state_id: str) -> Optional[WorkflowState]:
        """Get a workflow state by ID."""
        return self._states.get(state_id)
    
    def cancel_workflow(self, state_id: str) -> bool:
        """Cancel a running workflow."""
        state = self._states.get(state_id)
        if state and state.status == "running":
            state.status = "cancelled"
            state.end_time = datetime.utcnow()
            return True
        return False
    
    def list_workflows(self) -> List[str]:
        """List all registered workflow IDs."""
        return list(self._workflows.keys())
    
    def get_workflow_steps(self, workflow_id: str) -> Optional[List[WorkflowStep]]:
        """Get the steps for a workflow."""
        return self._workflows.get(workflow_id)
