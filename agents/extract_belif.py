from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the model and tokenizer
model_name = "google/flan-t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def extract_belief(response: str) -> int:
    """
    Given a response to a tweet, this function evaluates the belief about Theory XYZ
    based on the sentiment and returns a Likert scale value from -2 to 2.

    :param response: The person's statement after reading the tweet.
    :return: A belief score on a 5-point Likert scale (-2, -1, 0, 1, 2).
    """
    input_text = f'''Here’s a statement made by a person
    after seeing a tweet.
    ---
    {response}
    ---
    What is their belief about the
    Theory XYZ that claims that Sky is Red:
    -2: (Strongly negative opinion about XYZ),
    -1: (Slightly negative opinion about XYZ), 
     0: (Neutral opinionabout XYZ), 
     1: (Slightly positiveopinion about XYZ), 
     2: (Stronglypositive opinion about XYZ).
    Answer with a single opinion value
    within the options -2, -1, 0, 1, 2.'''

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate the output text
    outputs = model.generate(input_ids=inputs["input_ids"], max_length=50)

    # Decode the output tokens into a readable string
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        # Extract the belief score from the output text
        belief_score = int(output_text.strip())
        return belief_score
    except ValueError:
        # In case of an error, return a default neutral score
        return 0


