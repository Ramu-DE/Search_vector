# 🎬 AI Movie Search - START HERE

## 🚀 Quick Start (30 seconds)

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run the app
chainlit run chainlit_app.py -w

# 3. Open browser
# http://localhost:8000
```

That's it! Start chatting with your AI movie assistant.

---

## 💬 Try These in the Chat

- "I want an epic space adventure"
- "Dark psychological thriller"
- "Highly rated 90s drama about redemption"
- "Something like Inception but less confusing"

---

## ✅ What's Running

✓ **Chainlit Chat** - Conversational AI interface  
✓ **Claude Sonnet 4.6** - 1M context (via AWS Bedrock)  
✓ **Qdrant** - Vector database (15 movies)  
✓ **Bedrock Titan v2** - 1024-dim embeddings  

**No Anthropic API key needed** - uses AWS credentials

---

## 📚 Documentation

| Read This If... | Document |
|-----------------|----------|
| **Using the chat** | CHAINLIT_GUIDE.md |
| **Using the API** | api.py (FastAPI docs at /docs) |
| **Understanding the system** | FINAL_IMPLEMENTATION.md |
| **Deploying to production** | DEPLOYMENT.md |
| **Full technical details** | BEDROCK_CLAUDE_READY.md |

---

## 🎯 Features

✅ Natural language movie search  
✅ AI-powered query enhancement  
✅ Intelligent result summaries  
✅ 1M token context window  
✅ Real-time conversational interface  
✅ Multiple Claude model options  

---

## 🔧 Alternative Interfaces

### REST API
```bash
python api.py
# Access: http://localhost:8000/docs
```

### CLI Demo
```bash
python intelligent_search.py
```

---

## 💡 Commands

Type these in the chat:

- `/help` - Available commands
- `/models` - List Claude models
- `/examples` - Example queries

---

**Have fun searching!** 🎬
