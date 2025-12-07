# Task 4: Generative Text Model
# Instructions: Create a text generation model using GPT or LSTM to generate coherent paragraphs.
# Libraries: Hugging Face Transformers, PyTorch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt):
    """
    Generates a paragraph of text based on a user's input prompt using GPT-2.
    """
    print("Loading GPT-2 model... (this may take a moment)")
    
    # Load the pre-trained GPT-2 model and tokenizer
    # GPT-2 is excellent at predicting the next words in a sequence
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Encode the input prompt into numbers that the model understands
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    print("Generating text...")
    # Generate the output
    # max_length=150: The text will be about 150 words long
    # temperature=0.7: Controls creativity (0.7 is a good balance between creative and coherent)
    # no_repeat_ngram_size=2: Prevents the model from repeating the same phrases over and over
    outputs = model.generate(
        inputs, 
        max_length=200, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the numbers back into human-readable text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    print("\n--- AI Text Generator ---")
    
    # Get input from the user
    user_prompt = input("Enter a starting sentence for the AI (e.g., 'The future of AI is'): ")
    
    if user_prompt:
        generated_paragraph = generate_text(user_prompt)
        
        print("\n--- Generated Result ---")
        print(generated_paragraph)
        print("------------------------")
    else:
        print("Please enter a valid prompt.")