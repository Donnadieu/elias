# AI Assistant with RAG Capabilities

An advanced AI assistant built with AutoGen's AgentChat framework that combines code execution with Retrieval-Augmented Generation (RAG) for context-aware responses.

## Project Structure

- `agents/`
  - `personal_assistant_agent.py`: Main agent class with RAG capabilities
  - `rag_system.py`: RAG (Retrieval-Augmented Generation) system implementation
- `tools/`
  - `executor_tool.py`: Safe Python code execution in a restricted context
- `run.py`: Main entry point with CLI interface
- `requirements.txt`: Project dependencies
- `.env.example`: Example environment configuration

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your API keys (OpenAI and Qdrant)
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Prerequisites

- Python 3.8+
- OpenAI API key (for GPT models)
- Qdrant Cloud account (for vector storage)

## Usage

Start the assistant:

```bash
python run.py
```

### Basic Commands

- Ask any question and the assistant will use its knowledge base to respond
- Add information to the knowledge base: `add to knowledge: [information]`
- Type `exit` or `quit` to end the session

### Example Interactions

```
You: What is the capital of France?
Assistant: The capital of France is Paris.

You: add to knowledge: The Eiffel Tower is located in Paris.
Assistant: Document added to knowledge base with ID: abc123...

You: Where is the Eiffel Tower located?
Assistant: The Eiffel Tower is located in Paris.

You: Generate a Python function to sort a list of dictionaries by a key
Assistant: Here's a Python function that sorts a list of dictionaries by a specified key...
```

## RAG System

The assistant uses a Retrieval-Augmented Generation (RAG) system that:

1. Stores documents in a vector database (Qdrant)
2. Retrieves relevant context for each query
3. Generates responses using the retrieved context and a language model

### Adding Documents

You can add documents to the knowledge base in several ways:

1. **Interactive**: Use the `add to knowledge:` command
2. **Programmatic**: Use the `add_document()` method in Python
3. **Bulk Loading**: Create a script to load multiple documents at once

### Searching the Knowledge Base

The assistant automatically searches the knowledge base when answering questions. You can also search directly:

```python
results = await assistant.rag.search("your query", top_k=3)
```

## Security Considerations

- Code execution is performed in a restricted environment
- API keys should be kept secure and never committed to version control
- The knowledge base may store sensitive information - ensure proper access controls

## Future Enhancements

- [ ] Add document chunking for better retrieval
- [ ] Implement user authentication
- [ ] Add support for multiple knowledge bases
- [ ] Improve query understanding and retrieval quality
- [ ] Add web interface
- [ ] Implement conversation history

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT
