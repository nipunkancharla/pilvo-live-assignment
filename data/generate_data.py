import json
import random
from faker import Faker

fake = Faker()
# Set a seed for reproducibility
Faker.seed(42)
random.seed(42)

# --- Configuration ---
OUTPUT_TRAIN = "train.jsonl"
OUTPUT_DEV = "dev.jsonl"
NUM_TRAIN = 1000
NUM_DEV = 100

# Digits to words mapping for STT noise
DIGIT_MAP = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
}

def text_to_stt_noise(text):
    """
    Converts clean text to 'noisy' STT format.
    - Replaces digits with words (randomly).
    - Replaces special chars (@, .) with words.
    - Lowercases everything.
    """
    text = text.lower()
    
    # 1. Punctuation replacements
    replacements = {
        ".": " dot ",
        "@": " at ",
        "-": " ",
        "_": " ",
        ",": "",
        "?": "",
        "!": ""
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # 2. Number replacements (50% chance to spell out digits)
    noisy_chars = []
    for char in text:
        if char.isdigit() and random.random() > 0.3: # 70% chance to be a word
            noisy_chars.append(DIGIT_MAP.get(char, char))
            noisy_chars.append(" ") # Add space after number word
        else:
            noisy_chars.append(char)
            
    # Clean up multiple spaces
    return "".join(noisy_chars).strip().replace("  ", " ")

def generate_sample(utt_id):
    """
    Generates a single valid training example with 1-2 entities.
    """
    
    # We build the sentence in pieces to track indices perfectly
    sentence_parts = []
    entities = []
    
    # Templates: (Before Text, Entity Type, After Text)
    # We will dynamically generate the entity value
    templates = [
        (["my email is"], "EMAIL", ["please contact me"]),
        (["contact me at"], "EMAIL", [""]),
        (["the card number is"], "CREDIT_CARD", ["for payment"]),
        (["charge my card"], "CREDIT_CARD", [""]),
        (["call me at"], "PHONE", [""]),
        (["my phone number is"], "PHONE", ["thanks"]),
        (["i was born on"], "DATE", ["in the city"]),
        (["the date is"], "DATE", ["today"]),
        (["my name is"], "PERSON_NAME", ["and i live in"]),
        (["this is"], "PERSON_NAME", ["speaking"]),
        (["i am in"], "CITY", ["right now"]),
        (["traveling to"], "CITY", ["tomorrow"]),
        (["located at"], "LOCATION", ["near the bank"]),
    ]

    # Pick a random template structure
    # We might chain 2 templates together to make complex sentences
    num_segments = random.choice([1, 2])
    
    full_text = ""
    
    for _ in range(num_segments):
        before_list, label, after_list = random.choice(templates)
        
        # 1. Generate Raw Entity
        if label == "EMAIL":
            raw_entity = fake.email()
        elif label == "CREDIT_CARD":
            raw_entity = fake.credit_card_number()
        elif label == "PHONE":
            raw_entity = fake.phone_number()
        elif label == "PERSON_NAME":
            raw_entity = fake.name()
        elif label == "DATE":
            raw_entity = fake.date()
        elif label == "CITY":
            raw_entity = fake.city()
        elif label == "LOCATION":
            raw_entity = fake.address()

        # 2. Corrupt the parts independently
        before_text = text_to_stt_noise(random.choice(before_list))
        entity_text = text_to_stt_noise(raw_entity)
        after_text = text_to_stt_noise(random.choice(after_list))
        
        # 3. Stitch them together
        # Add a space if we are appending to existing text
        if full_text: 
            full_text += " and "
            
        # Append "Before" part
        if before_text:
            full_text += before_text + " "
            
        # RECORD START INDEX
        start_index = len(full_text)
        
        # Append Entity
        full_text += entity_text
        
        # RECORD END INDEX
        end_index = len(full_text)
        
        # Append "After" part
        if after_text:
            full_text += " " + after_text
            
        # Save the entity
        entities.append({
            "start": start_index,
            "end": end_index,
            "label": label
        })
        
    return {
        "id": utt_id,
        "text": full_text,
        "entities": entities
    }

def main():
    print(f"Generating {NUM_TRAIN} training samples...")
    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for i in range(NUM_TRAIN):
            sample = generate_sample(f"utt_train_{i:04d}")
            f.write(json.dumps(sample) + "\n")
            
    print(f"Generating {NUM_DEV} dev samples...")
    with open(OUTPUT_DEV, "w", encoding="utf-8") as f:
        for i in range(NUM_DEV):
            sample = generate_sample(f"utt_dev_{i:04d}")
            f.write(json.dumps(sample) + "\n")

    print("Done! Files saved to data/")

if __name__ == "__main__":
    main()