import json
import re
from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

OUTPUT_DIR = "phase1_outputs"
OUTPUT_FILE = "generated_prompts.json"

def ask_llm(client: InferenceClient, prompt: str) -> list[str]:
    """Sends a prompt to the HF API and extracts bulleted items."""
    messages = [
        {"role": "system", "content": "You are a helpful data generation assistant. Only output exactly what is requested, formatted as a bulleted list with hyphens (-). Do not include any conversational filler."},
        {"role": "user", "content": prompt}
    ]
    
    # Ping the serverless API
    response = client.chat_completion(
        messages=messages,
        max_tokens=1024,
        temperature=0.7
    )
    
    output_text = response.choices[0].message.content
    
    # Parse the output for bullet points
    lines = output_text.split('\n')
    results = [re.sub(r'^-\s*', '', line).strip() for line in lines if line.strip().startswith('-')]
    return results

def main():
    print("Connecting to Hugging Face Inference API...")
    # Using Llama 3 8B Instruct via the free serverless API
    client = InferenceClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct", 
        token=HF_TOKEN
    )

    dataset = {
        "standard_instructions": [],
        "paraphrases": [],
        "semantic_opposites": [],
        "ood_sentences": []
    }

    print("Generating Standard Instructions...")
    dataset["standard_instructions"] = ask_llm(client, "Generate 80 standard robotic manipulation instructions for a tabletop robot arm. Examples: 'pick up the red cube', 'put the white mug on the left plate and put the yellow and white mug on the right plate', 'move the plate forward'. Output each on a new line starting with a hyphen.")[:60]
    
    print("Generating Paraphrases...")
    dataset["paraphrases"] = ask_llm(client, "Generate 40 different ways to tell a robot to 'pick up the apple'. Use synonyms like grab, lift, grasp, take, take hold etc. Output each on a new line starting with a hyphen.")[:40]

    print("Generating Opposites...")
    dataset["semantic_opposites"] = ask_llm(client, "Generate 40 total robotic instructions (20 contrasting pairs) involving manipulating objects. "
        "These must be full sentences. For each pair, change only the directional or action word to its semantic opposite. "
        "Examples: '- move the red block to the left' followed by '- move the red block to the right', "
        "or '- open the main gripper completely' followed by '- close the main gripper completely'. "
        "Output each instruction on a new line starting with a hyphen.")[:40]

    print("Generating OOD Sentences...")
    dataset["ood_sentences"] = ask_llm(client, "Generate 40 random, conversational English sentences that have absolutely nothing to do with robots. Examples: 'The sky is blue', 'I like pizza'. Output each on a new line starting with a hyphen.")[:40]

    # Create folder if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build full path
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    # Save file inside phase1_outputs/
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Saved categorized prompts to '{output_path}'!")

if __name__ == "__main__":
    main()