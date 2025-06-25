import torch
import librosa
from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from jiwer import wer, mer, wil, cer


def transcribe_audio(model, processor, audio_path, device):
    """
    Transcribe mp3 audio using the loaded Whisper model
    """
    try:
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)

        input_features = processor.feature_extractor(
            audio_array, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return ""


def load_whisper_model(model_size):
    """
    Load a Whisper model of the specified size
    """
    model_name = f"openai/whisper-{model_size}"
    processor = WhisperProcessor.from_pretrained(
        model_name, cache_dir="/global/lynx_arm_esa/user/amisingh/temp"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, cache_dir="/global/lynx_arm_esa/user/amisingh/temp"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, processor, device


def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate BLEU score and F1 metrics for text generation evaluation.

    Args:
        predictions: List of predicted texts
        references: List of reference texts

    Returns:
        Dictionary containing BLEU, precision, recall, and F1 scores
    """
    metrics = {
        "wer": wer(references, predictions),  # Word Error Rate
        "mer": mer(references, predictions),  # Match Error Rate
        "wil": wil(references, predictions),  # Word Information Lost
    }

    metrics["cer"] = cer(references, predictions)

    try:
        ref_tokens = [ref.split() for ref in references]
        pred_tokens = [pred.split() for pred in predictions]

        weights = [
            (1.0, 0, 0, 0),  # BLEU-1
            (0.5, 0.5, 0, 0),  # BLEU-2
            (0.33, 0.33, 0.34, 0),  # BLEU-3
            (0.25, 0.25, 0.25, 0.25),
        ]  # BLEU-4

        smoothing = SmoothingFunction().method1
        for i, weight in enumerate(weights, 1):
            metrics[f"bleu_{i}"] = sentence_bleu(
                ref_tokens,  # All references for this prediction
                pred_tokens[0],
                weights=weight,
                smoothing_function=smoothing,
            )

        metrics["bleu"] = metrics["bleu_4"]

    except Exception as e:
        print(f"BLEU calculation failed: {e}")
        metrics["bleu"] = 0.0
        for i in range(1, 5):
            metrics[f"bleu_{i}"] = 0.0

    try:
        ref_tokens_flat = []
        pred_tokens_flat = []

        for ref in references:
            ref_tokens_flat.extend(ref.lower().split())
        for pred in predictions:
            pred_tokens_flat.extend(pred.lower().split())

        ref_set = set(ref_tokens_flat)
        pred_set = set(pred_tokens_flat)

        true_positives = len(ref_set.intersection(pred_set))

        precision = true_positives / len(pred_set) if pred_set else 0.0
        recall = true_positives / len(ref_set) if ref_set else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1_score"] = f1

        # Additional metrics
        metrics["true_positives"] = true_positives
        metrics["false_positives"] = len(pred_set - ref_set)
        metrics["false_negatives"] = len(ref_set - pred_set)

    except Exception as e:
        print(f"F1 calculation failed: {e}")
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1_score"] = 0.0

    return metrics


if __name__ == "__main__":
    original = "Hello, my name is Andrej Karpathy. I am the Director of AI at Tesla. I am also a professor at Stanford University. I have a PhD in computer science from Stanford University. I am also a co-founder of OpenAI. I am also a co-founder of the Andrej Karpathy Foundation."
    mapper = {}
    model_sizes = ["base.en", "tiny.en", "small.en", "medium.en", "large-v2"]
    for model_size in model_sizes:
        model, processor, device = load_whisper_model(model_size)
        prediction = transcribe_audio(model, processor, "GPT_Andrej.mp3", device)
        metrics = calculate_metrics([original], [prediction])
        print(f"Model: whisper-{model_size}")
        print(metrics)
        mapper[model_size] = metrics
    print(mapper)
