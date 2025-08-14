import argparse
import os
import numpy as np
import pandas as pd
import time
import requests
import json

from crop import crop

choices = ["A", "B", "C", "D"]


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def call_lmstudio_api(prompt, api_base, model, max_tokens=1, temperature=0):
    """Call LM Studio's OpenAI-compatible API"""
    url = f"{api_base}/v1/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": 5,  # LM Studio may not support 100 logprobs
        "echo": False  # LM Studio typically doesn't support echo
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

def eval(args, subject, model, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        # For simple evaluation without logprobs, we'll just get the completion
        # and check if it matches the correct answer
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            response = call_lmstudio_api(
                prompt + " ",  # Add space to encourage single letter response
                args.api_base,
                model,
                max_tokens=1,
                temperature=0
            )
            
            if response:
                break
            
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying... ({retry_count}/{max_retries})")
                time.sleep(2)
        
        if not response or 'choices' not in response or len(response['choices']) == 0:
            print(f"Failed to get response for question {i+1}")
            # Default to random guess
            pred = "A"
            probs = [0.25, 0.25, 0.25, 0.25]
        else:
            # Extract the predicted answer
            completion = response['choices'][0].get('text', '').strip().upper()
            
            # Try to extract the answer letter
            pred = None
            for ans in answers:
                if ans in completion:
                    pred = ans
                    break
            
            if pred is None:
                # If no valid answer found, default to A
                pred = "A"
            
            # Since LM Studio might not provide reliable logprobs,
            # we'll create a simple probability distribution
            probs = [0.1, 0.1, 0.1, 0.1]
            pred_idx = answers.index(pred) if pred in answers else 0
            probs[pred_idx] = 0.7  # Give high probability to predicted answer

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{test_df.shape[0]} questions completed")

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)
    
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    
    return cors, acc, all_probs

def main(args):
    # Use the model name directly
    model = args.model
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if args.subjects:
        # Allow running on specific subjects
        subjects = [s for s in subjects if s in args.subjects]
        print(f"Running on subjects: {subjects}")

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    results_dir = os.path.join(args.save_dir, f"results_lmstudio_{model.replace('/', '_')}")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    print(f"Model: {model}")
    print(f"API Base: {args.api_base}")
    print(f"Subjects: {len(subjects)}")
    print(f"N-train: {args.ntrain}")
    print()

    all_cors = []
    subject_accs = {}

    for subject in subjects:
        print(f"\nEvaluating: {subject}")
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

        cors, acc, probs = eval(args, subject, model, dev_df, test_df)
        all_cors.append(cors)
        subject_accs[subject] = acc

        # Save results
        test_df[f"{model}_correct"] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df[f"{model}_choice{choice}_probs"] = probs[:, j]
        test_df.to_csv(os.path.join(results_dir, f"{subject}.csv"), index=None)

    # Calculate and print overall results
    weighted_acc = np.mean(np.concatenate(all_cors))
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    for subject, acc in subject_accs.items():
        print(f"{subject}: {acc:.3f}")
    print("-"*50)
    print(f"Overall Average Accuracy: {weighted_acc:.3f}")
    
    # Save summary
    summary_df = pd.DataFrame(list(subject_accs.items()), columns=['Subject', 'Accuracy'])
    summary_df.loc[len(summary_df)] = ['Overall', weighted_acc]
    summary_df.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
    print(f"\nResults saved to {results_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model using LM Studio's API on MMLU benchmark")
    parser.add_argument("--ntrain", "-k", type=int, default=5,
                        help="Number of training examples for few-shot learning")
    parser.add_argument("--data_dir", "-d", type=str, default="data",
                        help="Directory containing the MMLU data")
    parser.add_argument("--save_dir", "-s", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--api_base", type=str, default="http://localhost:1234",
                        help="LM Studio API base URL (default: http://localhost:1234)")
    parser.add_argument("--model", "-m", type=str, default="local-model",
                        help="Model name to use (as configured in LM Studio)")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Specific subjects to evaluate (default: all)")
    args = parser.parse_args()
    main(args)