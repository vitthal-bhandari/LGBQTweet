import openai
import pandas as pd
from tqdm.notebook import tqdm
import os
import sys
import warnings

warnings.filterwarnings("ignore")

API_KEY = ''
openai.api_key = API_KEY
model_id = 'gpt-3.5-turbo'

def classify_text(prompt, max_tokens=2000):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "First go through the complete prompt i.e. paragraph, then perform accordingly. Paragraph starts from here: Sexual orientation and gender identity are complex and diverse aspects of human experience. Bisexuality refers to attraction to both males and females, while pansexuality includes attraction to people of any sex or gender identity. A person who identifies as gay or lesbian is attracted to individuals of the same sex. Transgender individuals have a gender identity that differs from the sex assigned to them at birth. Queer is an umbrella term used by some to describe individuals who identify as non-heterosexual and/or non-cisgender. Intersex individuals may have anatomical characteristics that are not typically male or female. Asexual individuals do not experience sexual attraction or desire. Understanding and respecting these various identities is crucial for promoting inclusivity and respecting individual autonomy. The homophobia definition is the fear, hatred, discomfort with, or mistrust of people who are lesbian, gay, or bisexual. Biphobia is fear, hatred, discomfort, or mistrust, specifically of people who are bisexual. Transphobia is fear, hatred, discomfort with, or mistrust of people who are transgender, genderqueer. Sentences should be coded as hateful when 1. Does the target (group) belong to one of the following groups: sexual identity, gender identity, and sexual orientation? NOTE: The target (group) must belong to non-dominant group. Does the text contain an explicit reference to the group (related to above specified target(s)) through a stereotype, group characteristic or slur or a direct reference to the target group itself? Does the tweet incite violence, hate, or discrimination or group insult? If it does, it should be labelled as hateful. Sentences should be coded as non-hateful when It refers to communication or anything that is respectful, considerate, and avoids any language or behavior that may cause harm or offense to individuals or groups based on their race, ethnicity, gender, sexual orientation, religion, or any other personal characteristic. It values diversity, promotes inclusivity, and fosters constructive dialogue and understanding among people with different perspectives and backgrounds. Non-hateful speech recognizes the importance of freedom of expression, but also understands the responsibility that comes with it, to use language and behavior that upholds human dignity and promotes positive relationships. There might be some words which will change the meaning of the sentence but the intention will be non-hateful, hope you got my point. For example, this sentnce 'Oh fuck I love being lesbian. Girls are so hot. #lesbiansquad' contains offensive language, but the intention behind the comtext is non-hateful. Now, you are a text annotation classifier who will classify the text as hateful or non-hateful. Just label the data, it must be hateful or non-hateful (only 1 word)"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        n=1,
        temperature=0.7
    )

    generated_text = response.choices[0].message.content.strip()
    return generated_text

def process_sentences(data, output_file='labeled_sentences.csv', batch_size=1, input_file='updated_input.csv'):
    if not os.path.exists(output_file):
        df = pd.DataFrame(columns=['text', 'label', 'reason'])
        df.to_csv(output_file, index=False)

    # Create a copy of the input DataFrame
    updated_input = data.copy()

    while not updated_input.empty:
        labeled_sentences = []
        for _, row in tqdm(updated_input.head(batch_size).iterrows(), file=sys.stdout):
            sentence = row['text']
            print(f"Processing sentence: {sentence}")
            
            # Prompt for classification label
            prompt_label = f"Classify the following sentence: \"{sentence}\""
            label = classify_text(prompt_label, max_tokens=50).strip()

            # Prompt for classification reason
            prompt_reason = f"Provide a reason for the {label} classification of the sentence: \"{sentence}\""
            reason = classify_text(prompt_reason, max_tokens=150).strip()

            labeled_sentences.append({'text': sentence, 'label': label, 'reason': reason})

            # Remove the processed sentence from the updated_input DataFrame
            updated_input = updated_input[updated_input['text'] != sentence].reset_index(drop=True)

        # Save the updated input DataFrame to a new file
        updated_input.to_csv(input_file, index=False)

        # Read existing CSV file
        df_existing = pd.read_csv(output_file)

        # Concatenate new labeled sentences with the existing ones
        df_new = pd.DataFrame(labeled_sentences)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

        # Remove duplicates
        df_combined.drop_duplicates(subset=['text'], inplace=True)
        df_combined.reset_index(drop=True, inplace=True)

        # Save the updated DataFrame to the CSV file
        df_combined.to_csv(output_file, index=False)

    # Check if the total row count is consistent
    original_row_count = len(data)
    updated_input_row_count = len(pd.read_csv(input_file))
    saved_output_row_count = len(pd.read_csv(output_file))

    assert original_row_count == updated_input_row_count + saved_output_row_count, "Row count mismatch detected."

def main(input_dataframe, output_file='labeled_sentences.csv', batch_size=1):
    print("Starting to process sentences...")
    process_sentences(input_dataframe, output_file=output_file, batch_size=batch_size)
    print("Finished processing sentences.")

if __name__ == "__main__":
    # Load the original or updated input DataFrame
    input_file = 'updated_input.csv'

    if os.path.exists(input_file):
        input_dataframe = pd.read_csv(input_file)
    else:
        input_dataframe = pd.read_csv('10k.csv')
        input_dataframe.reset_index(inplace=True)

    try:
        main(input_dataframe)
        print("Sentences have been labeled and saved to labeled_sentences.csv.")
    except Exception as e:
        print(f"Error occurred: {e}. Process stopped.")
