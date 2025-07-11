"""
Task Agent for the personal assistant using Supabase for task storage.
"""
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from supabase import create_client, Client

# Try to load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables directly.")


class TaskAgent:
    """
    Agent responsible for managing tasks in a Supabase PostgreSQL database.
    
    This agent handles storing, retrieving, and deleting tasks such as reminders,
    goals, and calendar items.
    """
    
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        """
        Initialize the TaskAgent with Supabase credentials.
        
        Args:
            supabase_url: The Supabase project URL. If None, uses SUPABASE_URL env var.
            supabase_key: The Supabase API key. If None, uses SUPABASE_KEY env var.
        
        Raises:
            ValueError: If Supabase credentials are not provided or found in environment.
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase configuration is incomplete. "
                "Please set SUPABASE_URL and SUPABASE_KEY environment variables."
            )
        
        self.client = create_client(self.supabase_url, self.supabase_key)
        self.table_name = "tasks"
    
    def add_task(self, text: str, due: Optional[datetime] = None, metadata: Dict[str, Any] = None) -> str:
        """
        Add a new task to the database.
        
        Args:
            text: The task description or content.
            due: Optional due date and time for the task.
            metadata: Additional information about the task (e.g., agent name, priority).
        
        Returns:
            The ID of the newly created task.
            
        Raises:
            Exception: If there's an error communicating with Supabase.
        """
        try:
            task_id = str(uuid.uuid4())
            task_data = {
                "id": task_id,
                "text": text,
                "due": due.isoformat() if due else None,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            response = self.client.table(self.table_name).insert(task_data).execute()
            
            if hasattr(response, 'error') and response.error:
                raise Exception(f"Failed to add task: {response.error.message}")
                
            return task_id
        except Exception as e:
            print(f"Error adding task: {str(e)}")
            raise
    
    def get_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve a list of tasks from the database.
        
        Args:
            limit: Maximum number of tasks to retrieve.
        
        Returns:
            List of task dictionaries.
            
        Raises:
            Exception: If there's an error communicating with Supabase.
        """
        try:
            response = self.client.table(self.table_name) \
                .select("*") \
                .order("due", desc=False) \
                .limit(limit) \
                .execute()
                
            if hasattr(response, 'error') and response.error:
                raise Exception(f"Failed to get tasks: {response.error.message}")
                
            return response.data
        except Exception as e:
            print(f"Error getting tasks: {str(e)}")
            return []
    
    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task from the database.
        
        Args:
            task_id: The unique identifier of the task to delete.
        
        Returns:
            True if deletion was successful, False otherwise.
            
        Raises:
            Exception: If there's an error communicating with Supabase.
        """
        try:
            response = self.client.table(self.table_name) \
                .delete() \
                .eq("id", task_id) \
                .execute()
                
            if hasattr(response, 'error') and response.error:
                raise Exception(f"Failed to delete task: {response.error.message}")
                
            # Check if any rows were affected
            return len(response.data) > 0
        except Exception as e:
            print(f"Error deleting task: {str(e)}")
            return False
    
    def get_due_tasks(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get tasks that are due now or in the past.
        
        Args:
            now: The current datetime to compare against. If None, uses current time.
        
        Returns:
            List of due task dictionaries.
            
        Raises:
            Exception: If there's an error communicating with Supabase.
        """
        if now is None:
            now = datetime.now()
            
        try:
            response = self.client.table(self.table_name) \
                .select("*") \
                .not_.is_("due", "null") \
                .lte("due", now.isoformat()) \
                .order("due", desc=False) \
                .execute()
                
            if hasattr(response, 'error') and response.error:
                raise Exception(f"Failed to get due tasks: {response.error.message}")
                
            return response.data
        except Exception as e:
            print(f"Error getting due tasks: {str(e)}")
            return []
    
    def summarize_tasks(self, limit: int = 5) -> str:
        """
        Create a human-readable summary of upcoming tasks.
        
        Args:
            limit: Maximum number of tasks to include in the summary.
        
        Returns:
            A formatted string summarizing the tasks.
        """
        try:
            tasks = self.get_tasks(limit=limit)
            
            if not tasks:
                return "You have no upcoming tasks."
            
            summary = f"You have {len(tasks)} upcoming task(s):\n\n"
            
            for i, task in enumerate(tasks, 1):
                due_str = ""
                if task.get("due"):
                    try:
                        due_date = datetime.fromisoformat(task["due"])
                        due_str = f" (Due: {due_date.strftime('%Y-%m-%d %H:%M')})"
                    except (ValueError, TypeError):
                        pass
                
                summary += f"{i}. {task['text']}{due_str}\n"
            
            return summary
        except Exception as e:
            return f"Unable to retrieve tasks: {str(e)}"
    
    def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing task.
        
        Args:
            task_id: The unique identifier of the task to update.
            updates: Dictionary of fields to update.
        
        Returns:
            True if update was successful, False otherwise.
            
        Raises:
            Exception: If there's an error communicating with Supabase.
        """
        try:
            # Ensure we're not trying to update the id
            if "id" in updates:
                del updates["id"]
                
            # Convert datetime objects to ISO strings
            if "due" in updates and isinstance(updates["due"], datetime):
                updates["due"] = updates["due"].isoformat()
                
            response = self.client.table(self.table_name) \
                .update(updates) \
                .eq("id", task_id) \
                .execute()
                
            if hasattr(response, 'error') and response.error:
                raise Exception(f"Failed to update task: {response.error.message}")
                
            # Check if any rows were affected
            return len(response.data) > 0
        except Exception as e:
            print(f"Error updating task: {str(e)}")
            return False
