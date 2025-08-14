# Running MMLU Benchmark with LM Studio

This guide explains how to run the MMLU (Measuring Massive Multitask Language Understanding) benchmark using LM Studio's local API.

## Prerequisites

1. **LM Studio**: Download and install from [https://lmstudio.ai/](https://lmstudio.ai/)
2. **Python packages**: Install required dependencies:
   ```bash
   pip install pandas numpy requests
   ```
3. **MMLU Dataset**: Already downloaded to the `data/` directory

## Setup LM Studio

1. **Start LM Studio** and load your desired model
2. **Start the local server**:
   - Go to the "Local Server" tab in LM Studio
   - Click "Start Server" 
   - Default port is 1234 (URL: http://localhost:1234)
   - Note the model name shown in LM Studio

## Running the Benchmark

### Basic Usage

Run evaluation on all 57 subjects:
```bash
python evaluate_lmstudio.py
```

### Command Line Options

- `--api_base`: LM Studio API URL (default: http://localhost:1234)
- `--model`: Model name as shown in LM Studio (default: local-model)
- `--ntrain`: Number of few-shot examples (default: 5)
- `--data_dir`: Path to MMLU data (default: data)
- `--save_dir`: Directory for results (default: results)
- `--subjects`: Specific subjects to test (optional)

### Examples

**Test with a specific model:**
```bash
python evaluate_lmstudio.py --model "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
```

**Test only specific subjects:**
```bash
python evaluate_lmstudio.py --subjects abstract_algebra anatomy astronomy
```

**Use different number of few-shot examples:**
```bash
python evaluate_lmstudio.py --ntrain 3
```

**Custom API endpoint:**
```bash
python evaluate_lmstudio.py --api_base http://localhost:8080
```

## Output

Results are saved in the `results/` directory:
- Individual subject results: `results/results_lmstudio_[model]/[subject].csv`
- Summary of all subjects: `results/results_lmstudio_[model]/summary.csv`

The script displays:
- Progress updates during evaluation
- Per-subject accuracy scores
- Overall average accuracy across all subjects

## Performance Tips

1. **Start with fewer subjects** to test your setup:
   ```bash
   python evaluate_lmstudio.py --subjects abstract_algebra
   ```

2. **Reduce few-shot examples** for faster evaluation:
   ```bash
   python evaluate_lmstudio.py --ntrain 2
   ```

3. **GPU acceleration**: Ensure LM Studio is using GPU if available

## Troubleshooting

- **Connection refused**: Make sure LM Studio server is running
- **Model not found**: Check the model name matches exactly what's shown in LM Studio
- **Slow performance**: Consider using a smaller model or reducing `--ntrain`
- **Out of memory**: Try a smaller model or reduce context length by lowering `--ntrain`

## Notes

- The script uses LM Studio's OpenAI-compatible API
- Logprobs support may be limited depending on the model
- Accuracy scores can be compared with the official leaderboard in the main README