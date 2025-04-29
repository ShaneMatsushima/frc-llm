import re
from pathlib import Path
from bs4 import BeautifulSoup
import PyPDF2
import json

def load_and_process_game_manuals(directory):
    """Process FRC game manual PDFs into clean text"""
    texts = []
    for pdf_path in Path(directory).glob("*.pdf"):
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join([page.extract_text() for page in reader.pages])
            # Clean up FRC-specific formatting
            text = re.sub(r'\[.*?\]', '', text)  # Remove section references
            text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
            texts.append(text)
    return texts

def process_forum_data(html_files):
    """Extract meaningful discussions from forum HTML"""
    discussions = []
    for html_file in html_files:
        with open(html_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            threads = []
            for thread in soup.find_all(class_='thread'):
                title = thread.find(class_='title').text.strip()
                posts = [p.text.strip() for p in thread.find_all(class_='post')]
                threads.append({'title': title, 'posts': posts})
            discussions.extend(threads)
    return discussions

def create_training_examples(data, context_length=512):
    """Create training examples in prompt-completion format"""
    examples = []
    for item in data:
        # Create strategy questions
        if "strategy" in item['type']:
            prompt = f"Given the game rules {item['rules_ref']}, what would be an effective strategy for {item['scenario']}?"
            completion = item['strategy_description']
            examples.append({"prompt": prompt, "completion": completion})
        
        # Create rule clarification questions
        elif "rule" in item['type']:
            prompt = f"Can you clarify rule {item['rule_number']} about {item['rule_topic']}?"
            completion = item['rule_explanation']
            examples.append({"prompt": prompt, "completion": completion})
    
    return examples