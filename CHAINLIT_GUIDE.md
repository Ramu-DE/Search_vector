# 🎬 Chainlit Chat Interface Guide

## Overview

The Chainlit interface provides a **conversational AI experience** for movie search, powered by Claude's 1M context window.

---

## 🚀 Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Run Chainlit
chainlit run chainlit_app.py -w

# Or use the launcher
./RUN_APP.sh
# Choose option 1
```

**Access**: http://localhost:8000

---

## 💬 Chat Interface Features

### 1. **Conversational Search**
Just chat naturally about what you want:

```
You: "I want an epic space adventure"

AI: 🤖 AI Analysis
    Only Interstellar truly matches your space adventure 
    request...
    
    📽️ Top Results:
    1. Interstellar (2014) ⭐ 8.6/10
    ...
```

### 2. **Context Awareness**
The AI remembers your conversation:

```
You: "I want a thriller"
AI: [Shows thriller recommendations]

You: "Something more psychological"
AI: [Refines to psychological thrillers]
```

### 3. **Query Enhancement**
See how AI expands your query:

```
Original: "epic space adventure"
Enhanced:
- vast intergalactic journey with stunning visuals
- science fiction blockbuster exploring the cosmos
- thrilling outer space odyssey with heroic characters
```

### 4. **AI Summaries**
Get intelligent explanations:

```
🤖 AI Analysis:
"These films share a common thread of questioning 
the nature of reality. Start with The Matrix for 
the most accessible entry point..."
```

---

## 📋 Commands

Type these in the chat:

- `/help` - Show available commands
- `/models` - List Claude models
- `/settings` - Current configuration
- `/examples` - Example queries

---

## 🎯 Example Conversations

### Example 1: Theme-Based Search

```
You: "I want an epic space adventure"

AI: 🔍 Analyzing your query...
    
    🤖 AI Analysis:
    Only Interstellar truly matches your space adventure 
    request. The others are acclaimed films but don't 
    involve space...
    
    📽️ Top 5 Results:
    
    1. Interstellar (2014)
       ⭐ 8.6/10 | 📊 Score: 0.247 | 🎭 Sci-Fi
       A team of explorers travel through a wormhole...
       🎬 Director: Christopher Nolan
```

### Example 2: Mood-Based Search

```
You: "I'm in the mood for something dark and psychological"

AI: 🤖 AI Analysis:
    The strongest matches are The Silence of the Lambs 
    and The Matrix. Start with Silence of the Lambs for 
    genuine psychological tension...
    
    📽️ Top Results:
    1. The Silence of the Lambs (1991) ⭐ 8.6/10
    2. The Matrix (1999) ⭐ 8.7/10
    ...
```

### Example 3: Preference-Based Search

```
You: "highly rated drama from the 90s about redemption"

AI: 🤖 AI Analysis:
    The Shawshank Redemption is clearly the standout! 
    It's a 90s drama with redemption at its core...
    
    📽️ Top Results:
    1. The Shawshank Redemption (1994) ⭐ 9.3/10
```

---

## ⚙️ Configuration

### Change Claude Model

Edit `chainlit_app.py`:

```python
# Line ~25
search_system = IntelligentMovieSearch(claude_model='opus-4.8')
# Options: 'sonnet-4.6', 'opus-4.8', 'haiku-4.5', 'fable-5'
```

### Adjust Results Count

Edit `chainlit_app.py`:

```python
# Line ~90
result = await cl.make_async(search.search)(
    query=query,
    k=10,  # Change this (default: 5)
    enhance_query=True,
    summarize=True
)
```

### Disable Features

```python
result = await cl.make_async(search.search)(
    query=query,
    k=5,
    enhance_query=False,  # Disable query enhancement
    summarize=False        # Disable AI summary
)
```

---

## 🎨 Customization

### Change Branding

Edit `.chainlit/config.toml`:

```toml
[UI]
name = "Your App Name"
description = "Your description"

[UI.theme.light.primary]
main = "#FF0000"  # Your color
```

### Add Custom CSS

Create `public/style.css`:

```css
/* Custom styles */
.message-content {
    font-size: 16px;
}
```

Update `.chainlit/config.toml`:

```toml
[UI]
custom_css = "/public/style.css"
```

---

## 📊 Chat Profiles

Switch between different Claude models:

1. Click the profile icon (top right)
2. Select:
   - **Sonnet 4.6** - Best balance (default)
   - **Opus 4.8** - Most capable
   - **Haiku 4.5** - Fastest

---

## 🚀 Production Deployment

### Run with Uvicorn

```bash
chainlit run chainlit_app.py --host 0.0.0.0 --port 8000
```

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t ai-movie-search .
docker run -p 8000:8000 \
  -e AWS_REGION=us-west-2 \
  -e QDRANT_URL=... \
  -e QDRANT_API_KEY=... \
  ai-movie-search
```

### Environment Variables

```bash
# Required
export AWS_REGION=us-west-2
export QDRANT_URL=https://your-cluster.qdrant.io
export QDRANT_API_KEY=your-key

# Optional
export CHAINLIT_AUTH_SECRET=your-secret-key
export CHAINLIT_URL=https://your-domain.com
```

---

## 🔒 Authentication

Add authentication to `.chainlit/config.toml`:

```toml
[project]
# Enable authentication
enable_telemetry = false

# Set auth secret
auth_secret = "your-secret-key"
```

Or use OAuth:

```python
# chainlit_app.py
import chainlit as cl

@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    # Custom auth logic
    return default_user
```

---

## 💡 Advanced Features

### File Upload

```python
@cl.on_message
async def main(message: cl.Message):
    # Handle file uploads
    if message.elements:
        for element in message.elements:
            if element.type == "file":
                # Process uploaded file
                content = element.content
```

### Actions

```python
# Add action buttons
actions = [
    cl.Action(name="more", value="show_more", label="Show More"),
    cl.Action(name="refine", value="refine", label="Refine Search"),
]

await cl.Message(
    content="Results...",
    actions=actions
).send()
```

### Streaming Responses

```python
# Stream AI responses
msg = cl.Message(content="")
await msg.send()

for chunk in stream_response():
    msg.content += chunk
    await msg.update()
```

---

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use different port
chainlit run chainlit_app.py --port 8001
```

### Chainlit Not Found

```bash
source .venv/bin/activate
pip install chainlit
```

### Connection Issues

```bash
# Check environment variables
echo $QDRANT_URL
echo $QDRANT_API_KEY

# Test backend
python -c "from intelligent_search import IntelligentMovieSearch; s = IntelligentMovieSearch()"
```

---

## 📈 Performance

### Latency
- Initial load: ~2-3s (system initialization)
- Per query: 5-7s (with AI enhancement + summary)
- Fast mode: 2-3s (disable enhancement)

### Optimization

```python
# Faster responses
result = await cl.make_async(search.search)(
    query=query,
    k=3,                    # Fewer results
    enhance_query=False,    # Skip enhancement
    summarize=True          # Keep summary
)

# Use faster model
search_system = IntelligentMovieSearch(claude_model='haiku-4.5')
```

---

## 🎯 Use Cases

### 1. Movie Discovery Platform
- Users chat naturally about preferences
- AI suggests personalized recommendations
- Conversational refinement

### 2. Customer Service Bot
- Answer movie-related questions
- Provide intelligent recommendations
- Handle follow-up queries

### 3. Research Assistant
- Search movie database semantically
- Get AI analysis of results
- Export recommendations

---

## 📚 Resources

- **Chainlit Docs**: https://docs.chainlit.io/
- **Chainlit GitHub**: https://github.com/Chainlit/chainlit
- **Examples**: https://docs.chainlit.io/examples/community

---

## ✅ Checklist

- [x] Chainlit installed
- [x] Chat interface created
- [x] Commands implemented
- [x] AI summaries working
- [x] Query enhancement integrated
- [x] Configuration file setup
- [x] Welcome message configured
- [ ] Deploy to production
- [ ] Add authentication (optional)
- [ ] Customize branding (optional)

---

**Enjoy your conversational AI movie search!** 🎬
