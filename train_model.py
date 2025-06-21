import os
import torch
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForCTC, TrainingArguments, Trainer, Wav2Vec2ForCTC
import evaluate
import re
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATASET_PATH = "./data/common_voice_id" 
MODEL_TO_FINETUNE = "indonesian-nlp/wav2vec2-base-indonesian" 
OUTPUT_DIR = "./models/fine_tuned_stt_model" 

print(f"Memuat dataset dari: {DATASET_PATH}")

try:
    dataset = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(DATASET_PATH, "train.tsv"),
            "validation": os.path.join(DATASET_PATH, "dev.tsv"),
            "test": os.path.join(DATASET_PATH, "test.tsv"),
        },
        sep="\t",
        column_names=["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accent", "locale", "segment"]
    )
    print("Dataset berhasil dimuat!")
    
    print("\n--- DEBUG: Beberapa sampel data mentah (sebelum filter) ---")
    for split_name in dataset.keys():
        if len(dataset[split_name]) > 0:
            print(f"Split: {split_name}")
            for i in range(min(3, len(dataset[split_name]))): 
                print(f"  Contoh {i}:")
                print(f"    path: '{dataset[split_name][i].get('path')}' (tipe: {type(dataset[split_name][i].get('path'))})")
                print(f"    sentence: '{dataset[split_name][i].get('sentence')}' (tipe: {type(dataset[split_name][i].get('sentence'))})")
                if 'client_id' in dataset[split_name][i]:
                    print(f"    client_id: '{dataset[split_name][i].get('client_id')}' (tipe: {type(dataset[split_name][i].get('client_id'))})")
        else:
            print(f"Split: {split_name} is empty before filtering!")
    print("--- Akhir DEBUG --- \n")


except Exception as e:
    print(f"Error memuat dataset: {e}")
    print(f"Pastikan folder Common Voice ID ada di '{DATASET_PATH}' dan berisi file .tsv dan folder 'clips'.")
    exit()

print("Memfilter entri dataset dengan nilai 'None' pada kolom 'path' atau 'sentence'...")
initial_train_count = len(dataset["train"])
initial_validation_count = len(dataset["validation"])
initial_test_count = len(dataset["test"])

def filter_none_paths_and_sentences(batch):
    return isinstance(batch["path"], str) and batch["path"] is not None and \
           isinstance(batch["sentence"], str) and batch["sentence"] is not None

dataset = dataset.filter(filter_none_paths_and_sentences)

print(f"Setelah filter None: Train={len(dataset['train'])} (dihapus {initial_train_count - len(dataset['train'])}), "
      f"Validation={len(dataset['validation'])} (dihapus {initial_validation_count - len(dataset['validation'])}), "
      f"Test={len(dataset['test'])} (dihapus {initial_test_count - len(dataset['test'])}).")

print("Memfilter entri dataset yang tidak memiliki file audio...")
def check_audio_exists(batch):
    audio_file_path = os.path.join(DATASET_PATH, "clips", batch["path"])
    return os.path.exists(audio_file_path)

dataset = dataset.filter(check_audio_exists)
print(f"Setelah filter audio: Train={len(dataset['train'])}, Validation={len(dataset['validation'])}, Test={len(dataset['test'])}.")

def prepare_paths(batch):
    if 'audio' not in batch:
        batch["audio"] = os.path.join(DATASET_PATH, "clips", batch["path"])
    return batch

dataset = dataset.map(prepare_paths)

dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000)) 
print("Kolom 'audio' telah diatur ulang ke format audio 16kHz.")

print(f"Memuat processor dan model dari: {MODEL_TO_FINETUNE}")
processor = AutoProcessor.from_pretrained(MODEL_TO_FINETUNE)

model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_TO_FINETUNE,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

if torch.cuda.is_available():
    model = model.to("cuda")
    print("Model dipindahkan ke GPU.")
else:
    print("GPU tidak terdeteksi, pelatihan akan menggunakan CPU (lebih lambat).")

model.config.ctc_zero_infinity = True
model.config.pad_token_id = processor.tokenizer.pad_token_id

chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\]' 
def remove_special_characters(batch):
    if 'sentence' in batch and isinstance(batch['sentence'], str):
        text = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
        batch["sentence"] = " ".join(text.split()).strip()
    return batch

print("Mempra-proses dataset (mengekstrak fitur audio dan tokenisasi teks)...")
def prepare_dataset(batch):
    audio_data = batch["audio"]
    if not isinstance(audio_data["array"], torch.Tensor):
        input_values = processor(audio_data["array"], sampling_rate=audio_data["sampling_rate"]).input_values[0]
    else:
        input_values = processor(audio_data["array"].numpy(), sampling_rate=audio_data["sampling_rate"]).input_values[0]

    batch["input_values"] = input_values
    batch["input_length"] = len(input_values)
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

dataset = dataset.map(remove_special_characters, num_proc=os.cpu_count() // 2 or 1)
dataset = dataset.map(
    prepare_dataset,
    remove_columns=["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accent", "locale", "segment"],
    num_proc=os.cpu_count() // 2 or 1,
    desc="Processing audio and text"
)
print("Dataset berhasil dipra-proses!")

MAX_INPUT_LENGTH = processor.feature_extractor.sampling_rate * 30
MIN_INPUT_LENGTH = processor.feature_extractor.sampling_rate * 1

print(f"Memfilter sampel audio: max {MAX_INPUT_LENGTH/16000:.1f}s, min {MIN_INPUT_LENGTH/16000:.1f}s...")
dataset["train"] = dataset["train"].filter(
    lambda x: x["input_length"] < MAX_INPUT_LENGTH and x["input_length"] > MIN_INPUT_LENGTH
)
dataset["validation"] = dataset["validation"].filter(
    lambda x: x["input_length"] < MAX_INPUT_LENGTH and x["input_length"] > MIN_INPUT_LENGTH
)
dataset["test"] = dataset["test"].filter(
    lambda x: x["input_length"] < MAX_INPUT_LENGTH and x["input_length"] > MIN_INPUT_LENGTH
)
print(f"Dataset train setelah filter panjang: {len(dataset['train'])} sampel")
print(f"Dataset validation setelah filter panjang: {len(dataset['validation'])} sampel")
print(f"Dataset test setelah filter panjang: {len(dataset['test'])} sampel")

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(processor=processor)

print("Memuat metrik evaluasi (WER - Word Error Rate)...")
wer_metric = evaluate.load("wer") 

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1) 
    
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False) 

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

print("Mengatur konfigurasi pelatihan...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    group_by_length=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=10,
    fp16=torch.cuda.is_available(),
    save_steps=1000,
    eval_steps=1000,
    logging_steps=1000,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
    dataloader_num_workers=os.cpu_count() // 2 if os.cpu_count() else 0,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    report_to=["tensorboard"],
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
)

print("Memulai pelatihan model...")
trainer.train()
print("Pelatihan selesai!")

os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Model akhir dan processor disimpan di: {OUTPUT_DIR}")