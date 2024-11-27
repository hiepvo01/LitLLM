import requests
import json

def test_llama32():
    url = "http://138.25.249.85:11434/api/generate"
    
    payload = {
        "model": "llama3.2",
        "prompt": "What makes you different from Llama 2?"
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # Process the streaming response
        full_response = ""
        for line in response.text.splitlines():
            if line.strip():
                data = json.loads(line)
                if 'response' in data:
                    full_response += data['response']
                    
        print("\nResponse:", full_response)
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_llama32()