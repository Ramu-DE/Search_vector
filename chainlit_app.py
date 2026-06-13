#!/usr/bin/env python3
"""
AI-Powered Movie Search - Chainlit Chat Interface
Conversational movie search with Claude (1M context)
"""

import chainlit as cl
from intelligent_search import IntelligentMovieSearch
from bedrock_claude import BedrockClaude
import json

# Global search instance
search_system = None


@cl.on_chat_start
async def start():
    """Initialize chat session"""

    # Welcome message
    await cl.Message(
        content="""# 🎬 Welcome to AI Movie Search!

I'm your intelligent movie recommendation assistant, powered by **Claude Sonnet 4.6** with 1M context window.

I can help you find movies by:
- **Natural language queries** ("epic space adventure")
- **Mood and themes** ("something dark and psychological")
- **Preferences** ("highly rated 90s drama about redemption")
- **Complex requests** ("mind-bending sci-fi that questions reality")

Just tell me what you're looking for, and I'll search through my database and explain why each recommendation matches your request!

---

**Try asking:**
- "I want an epic space adventure"
- "Dark psychological thriller"
- "Inspirational movies about hope"
- "Crime movies with complex plots"
""",
        author="AI Assistant"
    ).send()

    # Initialize search system
    try:
        msg = cl.Message(content="🤖 Initializing AI search system...")
        await msg.send()

        global search_system
        search_system = IntelligentMovieSearch(claude_model='sonnet-4.6')

        msg.content = "✅ System ready! Ask me anything about movies."
        await msg.update()

        # Store in user session
        cl.user_session.set("search", search_system)
        cl.user_session.set("model", "sonnet-4.6")

    except Exception as e:
        await cl.Message(
            content=f"❌ Failed to initialize: {e}\n\nPlease check your configuration.",
            author="System"
        ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages"""

    query = message.content.strip()

    if not query:
        await cl.Message(content="Please enter a search query!").send()
        return

    # Get search system from session
    search = cl.user_session.get("search")
    if not search:
        await cl.Message(content="❌ Search system not initialized. Please refresh.").send()
        return

    # Handle commands
    if query.startswith("/"):
        await handle_command(query)
        return

    # Show typing indicator
    async with cl.Step(name="Searching", type="run") as step:
        step.output = "🔍 Analyzing your query with AI..."

        try:
            # Perform search
            result = await cl.make_async(search.search)(
                query=query,
                k=5,
                enhance_query=True,
                summarize=True
            )

            step.output = f"Found {result['count']} results"

        except Exception as e:
            await cl.Message(content=f"❌ Search failed: {e}").send()
            return

    # Build response
    response_parts = []

    # AI Summary
    if result['summary']:
        response_parts.append("## 🤖 AI Analysis\n")
        response_parts.append(result['summary'])
        response_parts.append("\n")

    # Results
    response_parts.append(f"\n## 📽️ Top {min(5, len(result['results']))} Results\n")

    for i, movie in enumerate(result['results'][:5], 1):
        response_parts.append(f"\n### {i}. {movie['title']} ({movie['year']})")
        response_parts.append(f"**⭐ {movie['rating']}/10** | **📊 Score: {movie['score']:.3f}** | 🎭 {movie['genre']}")
        response_parts.append(f"\n{movie['plot']}\n")
        response_parts.append(f"🎬 Director: {movie['director']} | Cast: {movie['cast']}")
        response_parts.append("\n---\n")

    # Enhanced queries (collapsible)
    if result['enhanced_queries']:
        details = "\n**Original:** " + query
        details += "\n\n**AI-Enhanced Queries:**\n"
        for i, eq in enumerate(result['enhanced_queries'], 1):
            details += f"{i}. {eq}\n"

        response_parts.append(f"\n<details>\n<summary>🔄 Query Enhancement Details</summary>\n{details}\n</details>\n")

    # Send response
    await cl.Message(
        content="".join(response_parts),
        author="AI Assistant"
    ).send()

    # Suggest follow-up
    await cl.Message(
        content="💡 *Want to refine your search? Just tell me what you're looking for!*",
        author="Suggestions"
    ).send()


async def handle_command(command: str):
    """Handle special commands"""

    cmd = command.lower().strip()

    if cmd == "/help":
        help_text = """
## 🎯 Available Commands

- `/help` - Show this help message
- `/models` - List available Claude models
- `/settings` - Show current settings
- `/examples` - Show example queries

## 💡 Tips

- Be natural! Describe what you're in the mood for
- I understand preferences like "highly rated", "from the 90s", "action-packed"
- I can handle complex queries like "mind-bending sci-fi that questions reality"
- Ask follow-up questions to refine results
"""
        await cl.Message(content=help_text).send()

    elif cmd == "/models":
        models_text = """
## 🤖 Available Claude Models

**Current**: Claude Sonnet 4.6 (1M context)

**Available via Bedrock:**
- **Opus 4.8** - 1M context, most capable
- **Opus 4.7** - 1M context, very capable
- **Sonnet 4.6** - 1M context, best balance (default)
- **Haiku 4.5** - 200K context, fastest
- **Fable 5** - 1M context, cutting edge

*Model selection can be changed in the code settings.*
"""
        await cl.Message(content=models_text).send()

    elif cmd == "/settings":
        model = cl.user_session.get("model", "sonnet-4.6")
        settings_text = f"""
## ⚙️ Current Settings

- **Model**: {model}
- **Context Window**: 1M tokens
- **Query Enhancement**: Enabled
- **AI Summarization**: Enabled
- **Vector DB**: Qdrant
- **Embeddings**: Bedrock Titan v2 (1024-dim)
- **Results per query**: 5 movies
"""
        await cl.Message(content=settings_text).send()

    elif cmd == "/examples":
        examples_text = """
## 💡 Example Queries

Try asking:

**By Theme:**
- "epic space adventure with stunning visuals"
- "dark psychological thriller that messes with your mind"
- "inspirational story about overcoming obstacles"

**By Mood:**
- "I'm in the mood for something dark and psychological"
- "something uplifting and heartwarming"
- "intense action-packed thriller"

**By Preferences:**
- "highly rated drama from the 90s"
- "sci-fi movies from the 2010s with great ratings"
- "crime movies with complex, twisting plots"

**Complex Requests:**
- "movies like Inception but less confusing"
- "films that make you question reality"
- "Christopher Nolan style sci-fi"
"""
        await cl.Message(content=examples_text).send()

    else:
        await cl.Message(content=f"Unknown command: {command}\n\nType `/help` for available commands.").send()


@cl.on_settings_update
async def setup_agent(settings):
    """Handle settings updates"""
    # Could add model switching here
    await cl.Message(content="Settings updated!").send()


# Chat settings (optional)
@cl.set_chat_profiles
async def chat_profile():
    """Define chat profiles for different models"""
    return [
        cl.ChatProfile(
            name="Sonnet 4.6",
            markdown_description="**Best Balance** - 1M context, excellent performance",
            icon="⚡",
        ),
        cl.ChatProfile(
            name="Opus 4.8",
            markdown_description="**Most Capable** - 1M context, highest quality",
            icon="🧠",
        ),
        cl.ChatProfile(
            name="Haiku 4.5",
            markdown_description="**Fastest** - 200K context, quick responses",
            icon="⚡",
        ),
    ]


if __name__ == "__main__":
    # Run with: chainlit run chainlit_app.py
    pass
