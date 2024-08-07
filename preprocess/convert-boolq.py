import os
import json
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import time

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SYSTEM_PROMPT = """You are a helpful assistant designed to output JSON. The json looks like this: `{"statement": "The generated statement..."}`"""
TASK_PROMPT = """Convert the question posted below into a definitive statement.\nRULES: 1. Do not add any extra information into the generated statment.\n2. Do not add any extra negation to the generated statement if there isn't any negation in the question.\nEXAMPLES\n
Example 1: Question: "do new zealand and australia have the same flag", Statement: "new zealand and australia have the same flag"\nExample 2: Question: "do you need to fast for random glucose test", Statement: "you need to fast for a random glucose test"\nExample 3: Question: "can you have two jobs in south africa", Statement: "you can have two jobs in south africa"\nExample 4: Question: "does hot and sour soup have a lot of sodium", Statement: "hot and sour soup has a lot of sodium"\nExample 5: Question: "has any supreme court nominee not been confirmed", Statement: "Some Supreme Court nominees have not been confirmed"\nExample 6: Question: "are family dollar and dollar general the same company", Statement: "family dollar and dollar general are the same company"\n\nNow please generate a statement for the following question: """


# Create output cache if it doesn't exist
if not os.path.exists("output_cache.csv"):
        pd.DataFrame(columns=["input", "output"]).to_csv("output_cache.csv", index=False)
print("Reading cache...")
read_cache = pd.read_csv("output_cache.csv")
print(f"Read {len(read_cache)} rows from cache")
write_cache = []

client = OpenAI(api_key=OPENAI_API_KEY)

def is_valid_output(output):
    # Step 1: Ensure the output is a string
    if not isinstance(output, str):
        return False

    try:
        # Step 2: Validate JSON format
        data = json.loads(output)

        # Step 3: Check if the structure matches {"statement": "Some statement..."}
        if "statement" in data and isinstance(data["statement"], str):
            return True
        else:
            return False

    except json.JSONDecodeError:
        # The output is not a valid JSON string
        return False

def check_read_cache(text):
    if text in read_cache["input"].to_list():
        return read_cache[read_cache["input"] == text]["output"].values[0]
    else:
        return None

def get_openai_response(text):
    cached_output = check_read_cache(text)
    if cached_output is not None:
        print("Using cached output...", cached_output)
        return cached_output
    
    full_input = TASK_PROMPT + text
    response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    response_format={ "type": "json_object" },
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_input}
    ]
    )
    output = response.choices[0].message.content
    write_cache.append([text, output])
    return output

def write_cache_to_file():
    print("Writing cache to file...")
    wc = pd.DataFrame(write_cache, columns=["input", "output"])
    full_cache = pd.concat([read_cache, wc])
    full_cache.drop_duplicates(subset=["input"], inplace=True)
    full_cache.to_csv("output_cache.csv", index=False)

def main():
    dataset = load_dataset("boolq")
    
    
    train = []
    validation = []

    print("Processing training data...")
    for row in tqdm(dataset['train']):
        train.append([row['question'], row['passage'], row['answer']])
        
    # for row in tqdm(dataset['validation']):
    #     validation.append([row['question'], row['passage'], row['answer']])
        
    
    count = 0
    for row in tqdm(train):
        time.sleep(0.25)
        question = row[0]
        statement = get_openai_response(question)
        if is_valid_output(statement):
            statement = json.loads(statement)["statement"]
            count += 1
            print("Question: ", question)
            print("Statement: ", statement)
            print()
        else:
            print("Invalid output for question: ", question)
        # if count > 50:
        #     break
        
        if not count % 10:
            write_cache_to_file()
            

    write_cache_to_file()
        

    
    
if __name__ == "__main__":
    main()