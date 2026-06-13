# Claude Model Configuration

This project now supports Claude API models with automatic fallback to the best available model.

## 🎯 Model Preference Order

The system automatically tests and selects the best available model:

1. **Claude Mythos 5** (`claude-mythos-5`) - Most capable (requires Project Glasswing)
2. **Claude Opus 4.8** (`claude-opus-4-8`) - Default fallback
3. **Claude Sonnet 4.6** (`claude-sonnet-4-6`) - Best speed/intelligence balance

## 🔧 Setup Instructions

### Step 1: Get an Anthropic API Key

1. Visit: https://console.anthropic.com/settings/keys
2. Create a new API key
3. Copy the key (starts with `sk-ant-`)

### Step 2: Configure Environment

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and add your key:

```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Step 3: Install Dependencies

```bash
pip install anthropic
# or if using virtual environment:
pip install -r requirements.txt
```

### Step 4: Test Your Configuration

```bash
python test_claude_model.py
```

This will:
- ✓ Test access to Claude Mythos 5 (if you have Project Glasswing access)
- ✓ Fall back to Claude Opus 4.8 if Mythos 5 is unavailable
- ✓ Display the selected default model

## 📊 What You'll See

### If You Have Mythos 5 Access:
```
🔍 Testing Claude model access...
   Testing claude-mythos-5... ✓ Available

✓ Default model set to: claude-mythos-5
```

### If You Don't Have Mythos 5 Access:
```
🔍 Testing Claude model access...
   Testing claude-mythos-5... ✗ Not available
   Testing claude-opus-4-8... ✓ Available

✓ Default model set to: claude-opus-4-8
```

## 💻 Usage in Your Code

```python
from model_config import ClaudeModelConfig

# Get the default model (automatically selected)
model = ClaudeModelConfig.get_model()
print(f"Using: {model}")

# Get authenticated client
client = ClaudeModelConfig.get_client()

# Make API call
response = client.messages.create(
    model=model,
    max_tokens=16000,
    messages=[{"role": "user", "content": "Your query here"}]
)
```

### Override Default Model

```python
# Use a specific model regardless of default
specific_model = ClaudeModelConfig.get_model(override="claude-sonnet-4-6")
```

## 🎫 Getting Claude Mythos 5 Access

Claude Mythos 5 is available exclusively through **Project Glasswing**.

**Note**: If you don't have Glasswing access, the system automatically uses **Claude Opus 4.8**, which is the most capable widely available model.

## 📝 Model Comparison

| Model | Context | Max Output | Pricing (per 1M tokens) | Use Case |
|-------|---------|------------|-------------------------|----------|
| **Mythos 5** | 1M | 128K | $10/$50 | Most demanding reasoning |
| **Opus 4.8** | 1M | 128K | $5/$25 | Default recommendation |
| **Sonnet 4.6** | 1M | 64K | $3/$15 | Production workloads |

## 🔍 Troubleshooting

### "No ANTHROPIC_API_KEY found"
- Make sure you created `.env` file
- Make sure the key is properly formatted: `ANTHROPIC_API_KEY=sk-ant-...`
- Try: `export ANTHROPIC_API_KEY='your-key'` in your terminal

### "Model not available"
- **Mythos 5**: Requires Project Glasswing participation
- **Opus/Sonnet**: Should work with any valid API key
- Check your API key permissions at: https://console.anthropic.com/

### Import Error
```bash
pip install anthropic
# or
pip install --upgrade anthropic
```

## 🚀 Next Steps

1. ✅ Set up your API key
2. ✅ Run `python test_claude_model.py`
3. ✅ Integrate into your search application
4. ✅ Check if you have Mythos 5 access (optional)

---

**Note**: This configuration is separate from your AWS Bedrock embedding models. The Claude API is used for advanced reasoning tasks, while Bedrock continues to handle embeddings.
