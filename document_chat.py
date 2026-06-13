#!/usr/bin/env python3
"""
Document Chat - Chainlit Interface for PDF Q&A
Chat with your documents using Claude (1M context)
"""

import chainlit as cl
from search_documents import DocumentSearch

# Global search instance
doc_search = None


@cl.on_chat_start
async def start():
    """Initialize chat session"""

    # Welcome message
    await cl.Message(
        content="""# 📄 Welcome to Document Chat!

I'm your AI document assistant, powered by **Claude Sonnet 4.6** with 1M context window.

I can help you find information in your uploaded documents by:
- **Answering questions** based on document content
- **Citing sources** with page numbers
- **Explaining complex topics** from the documents
- **Summarizing key information**

---

**Currently Available:**
✓ **NVIDIA.pdf** - NVIDIA DGX Spark documentation (35 pages, 38 chunks)

**Try asking:**
- "What is NVIDIA DGX Spark?"
- "What are the specs of DGX Spark?"
- "Tell me about AI development challenges"
- "What GPU does DGX Spark use?"
- "How much does NVIDIA AI Enterprise support cost?"
""",
        author="AI Assistant"
    ).send()

    # Initialize search system
    try:
        msg = cl.Message(content="🤖 Initializing document search system...")
        await msg.send()

        global doc_search
        doc_search = DocumentSearch(collection_name='documents', claude_model='sonnet-4.6')

        msg.content = "✅ System ready! Ask me anything about the documents."
        await msg.update()

        # Store in user session
        cl.user_session.set("search", doc_search)

    except Exception as e:
        await cl.Message(
            content=f"❌ Failed to initialize: {e}\n\nPlease check your configuration.",
            author="System"
        ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages"""

    question = message.content.strip()

    if not question:
        await cl.Message(content="Please ask a question!").send()
        return

    # Get search system from session
    search = cl.user_session.get("search")
    if not search:
        await cl.Message(content="❌ Search system not initialized. Please refresh.").send()
        return

    # Handle commands
    if question.startswith("/"):
        await handle_command(question)
        return

    # Show processing steps
    async with cl.Step(name="Searching", type="run") as step:
        step.output = "🔍 Searching documents..."

        try:
            # Search and answer
            result = await cl.make_async(search.answer_question)(question, k=5)
            step.output = f"Found {len(result['sources'])} relevant sources"

        except Exception as e:
            await cl.Message(content=f"❌ Search failed: {e}").send()
            return

    # Build response
    response_parts = []

    # Answer
    response_parts.append("## 📝 Answer\n")
    response_parts.append(result['answer'])
    response_parts.append("\n")

    # Sources
    response_parts.append(f"\n## 📚 Sources ({len(result['sources'])} documents)\n")

    for i, source in enumerate(result['sources'], 1):
        score_stars = "⭐" * min(5, int(source['score'] * 5))
        response_parts.append(
            f"\n**{i}. {source['source']} - Page {source['page']}** "
            f"({score_stars} {source['score']:.3f})"
        )
        response_parts.append(f"\n> {source['text']}\n")

    # Send response
    await cl.Message(
        content="".join(response_parts),
        author="AI Assistant"
    ).send()

    # Suggest follow-up
    await cl.Message(
        content="💡 *Ask follow-up questions to dig deeper into the documents!*",
        author="Suggestions"
    ).send()


async def handle_command(command: str):
    """Handle special commands"""

    cmd = command.lower().strip()

    if cmd == "/help":
        help_text = """
## 🎯 Available Commands

- `/help` - Show this help message
- `/docs` - List available documents
- `/examples` - Show example questions

## 💡 Tips

- Ask specific questions about document content
- I'll cite sources with page numbers
- Ask follow-up questions for more details
- Questions can be natural and conversational
"""
        await cl.Message(content=help_text).send()

    elif cmd == "/docs":
        docs_text = """
## 📚 Available Documents

**Currently Loaded:**

1. **NVIDIA.pdf** (35 pages, 38 chunks)
   - Topic: NVIDIA DGX Spark - AI Development Platform
   - Authors: Somnath Jana & Sagar Desai (NVIDIA)
   - Content: Hardware specs, software stack, use cases

**To add more documents:**
```bash
python ingest_pdf.py path/to/your.pdf documents
```
"""
        await cl.Message(content=docs_text).send()

    elif cmd == "/examples":
        examples_text = """
## 💡 Example Questions

**Technical Specifications:**
- "What are the specs of NVIDIA DGX Spark?"
- "What GPU does it use?"
- "How much memory does it have?"

**Use Cases:**
- "What is DGX Spark used for?"
- "Tell me about the NASDAQ use case"
- "What kind of AI workloads can it handle?"

**Software & Support:**
- "What software comes with DGX Spark?"
- "How much does NVIDIA AI Enterprise support cost?"
- "What is the DGX OS?"

**General Questions:**
- "What challenges does local AI development face?"
- "How does DGX Spark compare to cloud development?"
- "What is the evolution of AI from perception to physical AI?"
"""
        await cl.Message(content=examples_text).send()

    else:
        await cl.Message(content=f"Unknown command: {command}\n\nType `/help` for available commands.").send()


# Chat profiles for different models
@cl.set_chat_profiles
async def chat_profile():
    """Define chat profiles"""
    return [
        cl.ChatProfile(
            name="Sonnet 4.6",
            markdown_description="**Best Balance** - 1M context, excellent for documents",
            icon="📄",
        ),
        cl.ChatProfile(
            name="Opus 4.8",
            markdown_description="**Most Capable** - 1M context, deepest analysis",
            icon="🧠",
        ),
        cl.ChatProfile(
            name="Haiku 4.5",
            markdown_description="**Fastest** - 200K context, quick answers",
            icon="⚡",
        ),
    ]


if __name__ == "__main__":
    # Run with: chainlit run document_chat.py -w
    pass
