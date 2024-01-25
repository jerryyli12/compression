import argparse
from datasets import load_dataset
import os
import tqdm
import random
import numpy as np

from dotenv import load_dotenv
from genai.credentials import Credentials
from genai.model import Model
from genai.schemas import GenerateParams

EVAL_PROMPT = """Given a piece of text, an instruction for this text, and two AI assistant answers, your task is to choose the better answer and provide reasons. Evaluate the answers holistically, paying special attention to whether the response (1) follows the given instruction and (2) is
correct. If both answers correctly respond to the prompt, you should judge it as a tie.

Example 1:
'''
Text: We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformerbased model pre-trained to predict the next token in a document. The post-training alignment process results in improved performance on measures of factuality and adherence to desired behavior. A core component of this project was developing infrastructure and optimization methods that behave predictably across a wide range of scales. This allowed us to accurately predict some aspects of GPT-4's performance based on models trained with no more than 1/1,000th the compute of GPT-4.
Prompt: What is GPT4?
Assistant A: GPT4 is a large-scale language-trained transformer-based model.
Assistant B: GPT4 can produce outputs.
'''
Your output should be:
'''
{"reason": "The instruction asks what GPT4 is, and from the original text, we know that GPT4 is a multimodal, large-scale model that can generate text. Therefore, Assistant A is the closer answer, while Assistant B did not follow the instruction well in providing a response.", "choice": "A"}
'''
Example 2:
'''
Text: Making language models bigger does not inherently make them better at following a user's intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning. We then collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback. We call the resulting models InstructGPT. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, our results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.
Prompt: Write a title for the above text.
Assistant A: Improving Fine-Tuning for Language Models: A GPT-3-inspired Approach
Assistant B: Training language models to follow instructions with human feedback
'''
Your output should be:
'''
{"reason": "This text discusses how to make large language models follow
user instructions better, and Assistant B's response is more in line
with the meaning of the text.", "choice": "B"}
'''
Example 3:
'''
Text: Though technology companies announced massive layoffs last year, 2023 has been much worse. Layoffs have far outpaced last year's cuts, as tech giants including Amazon, Facebook parent company Meta, Microsoft, Google, IBM, SAP, and Salesforce - as well as many smaller companies - announce sweeping job cuts. The problem: Big Tech went on a hiring binge during the pandemic when lockdowns sparked a tech buying spree to support remote work and an uptick in e-commerce, and now they face revenue declines. Although global IT spending is forecast to rise in 2023, with enterprise software and IT services experiencing the greatest growth, the overall increase is expected to be modest, with data center systems and communications services growing by less than 1%, according to market research firm Gartner. Meanwhile hardware sales are forecast to decline. Continuing supply chain issues, inflation, and the war in Ukraine are also having an impact on both business and consumer spending, leading to fears of recession. According to data compiled by Layoffs.fyi, the online tracker keeping tabs on job losses in the technology sector, 784 tech companies have laid off about 208,181 staff so far this year, compared to 164,411 layoffs last year.
Prompt: How many job losses have occurred in the technology sector so far in 2023?
Assistant A: 208,181 job losses have occurred in the technology sector so far in 2023.
Assistant B: 208,181.
'''
Your output should be:
'''
{"reason": "Both answers are acceptable and correct. They should be a tie.", "choice": "Tie"}
'''
Your response should only be in the JSON format above; THERE SHOULD BE NO OTHER CONTENT INCLUDED IN YOUR RESPONSE. Write the "reason" key before writing the "choice" key, so that you think step-by-step before making your decision. KEEP YOUR REASONING BRIEF. Again, don't favor either A or B if they are both acceptable and correct -- judge a tie instead.
"""

class BatchIterator:
    def __init__(self, data, batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i:i + self.batch_size]

    def __len__(self):
        return np.ceil(len(self.data) / self.batch_size)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file', type=str, default='output.jsonl')
    parser.add_argument('--model', type=str, default='meta-llama/llama-2-70b-chat')

    args = parser.parse_args()
    print(args)

    dataset = load_dataset('json', data_files=args.file, split='train')
    print(dataset)

    load_dotenv()
    api_key = os.getenv("GENAI_KEY", None)
    api_url = os.getenv("GENAI_API", None)
    creds = Credentials(api_key, api_endpoint=api_url)
    params = GenerateParams(decoding_method='greedy', max_new_tokens=500)

    model = Model(args.model, params=params, credentials=creds)

    def get_prompt(data_point):
        if random.random() < 0.5:
            return f"{EVAL_PROMPT}\nText: {data_point['input']}\nPrompt: {data_point['prompt']}\nAssistant A: {data_point['uncompressed_answer']}\nAssistant B: {data_point['compressed_answer']}\n'''\nOutput: ", "A"
        return f"{EVAL_PROMPT}\nText: {data_point['input']}\nPrompt: {data_point['prompt']}\nAssistant A: {data_point['compressed_answer']}\nAssistant B: {data_point['uncompressed_answer']}\n'''\nOutput: ", "B"

    all_prompts = []
    prompt_to_data = {}
    for data_point in dataset:
        prompt, unc_assistant = get_prompt(data_point)
        data_point['unc_assistant'] = unc_assistant

        prompt_to_data[prompt] = data_point
        all_prompts.append(prompt)

    batching_inputs = tqdm.tqdm(BatchIterator(all_prompts, 10), desc="Batched inference")

    unc_wins, c_wins, ties = 0, 0, 0
    for _, batched_inputs in enumerate(batching_inputs):
        prompts = [prompt for prompt in batched_inputs]

        for i, result in enumerate(model.generate_async(prompts)):
            if result is not None:
                p = result.input_text
                g = result.generated_text

                data_point = prompt_to_data[p]
                try:
                    o = eval(g.strip())
                    if o['choice'] in 'AB':
                        if o['choice'] == data_point['unc_assistant']:
                            unc_wins += 1
                        else:
                            c_wins += 1
                    elif o['choice'] == 'Tie':
                        ties += 1
                    else:
                        print('Yikes!')
                except Exception as e:
                    print(e)
            else:
                print("oops")

    total = c_wins + ties + unc_wins
    print(f"Compressed context W/T/L: {c_wins / total}/{ties / total}/{unc_wins / total}")

if __name__ == "__main__":
    main()