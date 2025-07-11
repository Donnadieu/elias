-- Migration: Create tasks table for TaskAgent
-- Description: Creates a table to store tasks with due dates and metadata

-- Create extension for UUID generation if not exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text TEXT NOT NULL,
    due TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create index on due date for efficient querying of upcoming tasks
CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks (due);

-- Create index on created_at for sorting by creation time
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks (created_at);

-- Add comment to table
COMMENT ON TABLE tasks IS 'Stores user tasks, reminders, and calendar items';

-- Grant permissions (adjust as needed for your Supabase setup)
-- ALTER TABLE tasks ENABLE ROW LEVEL SECURITY;

-- Create policy for authenticated users (customize based on your auth requirements)
-- CREATE POLICY "Users can view their own tasks" ON tasks
--     FOR SELECT USING (auth.uid() = metadata->>'user_id');

-- CREATE POLICY "Users can insert their own tasks" ON tasks
--     FOR INSERT WITH CHECK (auth.uid() = metadata->>'user_id');

-- CREATE POLICY "Users can update their own tasks" ON tasks
--     FOR UPDATE USING (auth.uid() = metadata->>'user_id');

-- CREATE POLICY "Users can delete their own tasks" ON tasks
--     FOR DELETE USING (auth.uid() = metadata->>'user_id');
