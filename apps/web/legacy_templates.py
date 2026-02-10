"""HTML template for the legacy (v1) web interface.

Why: Keeps the old UI separate so it can be maintained or removed
independently of the enhanced version.
"""

LEGACY_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MultiHop Agent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; color: white; margin-bottom: 40px; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
        .header p { font-size: 1.1em; opacity: 0.9; }
        .main-content { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px; }
        @media (max-width: 768px) { .main-content { grid-template-columns: 1fr; } }
        .card { background: white; border-radius: 15px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        .input-section h2, .output-section h2 { color: #667eea; margin-bottom: 20px; font-size: 1.5em; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
        textarea { width: 100%; min-height: 120px; padding: 15px; border: 2px solid #ddd; border-radius: 8px; font-size: 1em; font-family: inherit; resize: vertical; transition: border-color 0.3s; }
        textarea:focus { outline: none; border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1); }
        button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 15px 40px; font-size: 1.1em; font-weight: 600; border-radius: 8px; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; width: 100%; }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3); }
        button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .loading { display: inline-block; width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite; margin-left: 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .reasoning-steps { background: #f8f9fa; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .reasoning-steps h3 { color: #667eea; margin-bottom: 15px; font-size: 1.2em; }
        .step { padding: 10px 15px; margin-bottom: 10px; background: white; border-left: 4px solid #667eea; border-radius: 4px; font-size: 0.95em; line-height: 1.6; }
        .answer-box { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 25px; border-radius: 10px; margin-top: 20px; }
        .answer-box h3 { margin-bottom: 15px; font-size: 1.3em; }
        .answer-content { font-size: 1.1em; line-height: 1.7; white-space: pre-wrap; }
        .empty-state { text-align: center; color: #999; padding: 40px; font-style: italic; }
        .history-section { margin-top: 30px; }
        .history-item { background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 3px solid #764ba2; }
        .history-question { font-weight: 600; color: #333; margin-bottom: 5px; }
        .history-answer { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header"><h1>MultiHop Agent</h1><p>LLM-based Q&A</p></div>
        <div class="main-content">
            <div class="card input-section">
                <h2>Ask</h2>
                <form method="POST" action="/ask">
                    <div class="form-group"><label for="question">Your question:</label><textarea id="question" name="question" placeholder="e.g. Where did Einstein work?" required></textarea></div>
                    <button type="submit">Submit</button>
                </form>
            </div>
            <div class="card output-section">
                <h2>Answer</h2>
                {% if reasoning_steps %}<div class="reasoning-steps"><h3>Reasoning</h3>{% for step in reasoning_steps %}<div class="step">{{ step }}</div>{% endfor %}</div>{% endif %}
                {% if answer %}<div class="answer-box"><h3>Final Answer</h3><div class="answer-content">{{ answer }}</div></div>{% endif %}
                {% if not answer and not reasoning_steps %}<div class="empty-state">Enter a question and submit...</div>{% endif %}
            </div>
        </div>
        {% if history %}<div class="card history-section"><h2>History</h2>{% for item in history %}<div class="history-item"><div class="history-question">Q: {{ item.question }}</div><div class="history-answer">A: {{ item.answer[:100] }}...</div></div>{% endfor %}</div>{% endif %}
    </div>
</body>
</html>
"""
