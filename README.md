## 🧠 Introduction

This project focuses on **English-to-Hindi Neural Machine Translation (NMT)** using the **Hugging Face Transformers** library. The goal is to fine-tune a pre-trained **MarianMT model** (`Helsinki-NLP/opus-mt-en-hi`) on the **IIT Bombay English–Hindi Parallel Corpus**, enabling high-quality translation between English and Hindi text.

Machine Translation is a key Natural Language Processing (NLP) task that enables communication across language barriers by automatically converting text from one language to another. Traditional rule-based systems struggle with linguistic complexity, idioms, and contextual nuances — whereas **Neural Machine Translation (NMT)** models leverage **deep learning**, particularly **Seq2Seq architectures with attention**, to generate fluent and context-aware translations.

---

## 📊 Project Overview

### 🔹 Objectives

* Fine-tune the **MarianMT** model for English→Hindi translation.
* Evaluate model performance using **SacreBLEU**.
* Deploy a simple and interactive **Gradio web interface** for real-time translation.

### 🔹 Key Features

✅ Fine-tuning a pre-trained Transformer model
✅ Custom preprocessing and tokenization pipeline
✅ Evaluation with BLEU metric using `evaluate`
✅ Freezing layers for faster and efficient training
✅ Interactive Gradio demo for live translation

---

## 🧩 Dataset

**Dataset:** [IIT Bombay English–Hindi Parallel Corpus (cfilt/iitb-english-hindi)](https://huggingface.co/datasets/cfilt/iitb-english-hindi)

* **Train Samples:** 1.6M+
* **Validation Samples:** 520
* **Test Samples:** 2,507
* **Languages:** English ↔ Hindi

This dataset provides high-quality, human-translated sentence pairs for English–Hindi machine translation tasks.

---

## ⚙️ Model and Libraries

### 🔹 Pretrained Model

`Helsinki-NLP/opus-mt-en-hi` (MarianMT model by Hugging Face)

### 🔹 Main Libraries

* 🤗 **Transformers**
* 🧮 **Datasets**
* 🧠 **PyTorch**
* 📈 **Evaluate (SacreBLEU)**
* ⚡ **Accelerate**
* 💬 **Gradio**

---

## 🧰 Installation

Run the following in your notebook or Colab environment:

```bash
!pip install datasets transformers "transformers[torch]" sentencepiece sacrebleu evaluate accelerate gradio sacremoses
```

---

## 🧾 Training Workflow

1. **Load Dataset**

   ```python
   from datasets import load_dataset
   dataset = load_dataset("cfilt/iitb-english-hindi")
   ```

2. **Load Tokenizer and Model**

   ```python
   from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
   tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
   model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
   ```

3. **Preprocessing and Tokenization**
   Tokenize English inputs and Hindi targets with truncation and padding.

4. **Fine-tuning**
   Fine-tune the MarianMT model using `Seq2SeqTrainer`:

   ```python
   from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
   trainer = Seq2SeqTrainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_datasets_train,
       eval_dataset=tokenized_datasets_validation,
       tokenizer=tokenizer,
       compute_metrics=compute_metrics,
   )
   trainer.train()
   ```

5. **Evaluation**
   Compute BLEU score using:

   ```python
   import evaluate
   metric = evaluate.load("sacrebleu")
   ```

6. **Inference (Translation)**

   ```python
   text = "Education is the foundation of progress."
   inputs = tokenizer(text, return_tensors="pt").to(device)
   translated = model.generate(**inputs, max_length=256)
   print(tokenizer.batch_decode(translated, skip_special_tokens=True))
   ```

---

## 💻 Deployment (Gradio Interface)

```python
import gradio as gr

def translate_fn(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    translated_tokens = model.generate(**inputs, max_length=256)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

demo = gr.Interface(fn=translate_fn, inputs="text", outputs="text", title="English → Hindi Translator")
demo.launch(share=True)
```

✨ Once launched, you’ll get a public URL to use the translator interactively.

---

## 📈 Results

| Metric                | Score                                |
| :-------------------- | :----------------------------------- |
| **BLEU (Validation)** | ~30–40 (varies by fine-tuning setup) |

The model produces coherent Hindi translations for most English sentences, maintaining grammatical structure and semantic context.

---

## 🏁 Conclusion

This project demonstrates how **Transformer-based NMT models** can be fine-tuned to achieve effective bilingual translation performance. Using **Hugging Face Transformers**, **MarianMT**, and the **IIT Bombay English–Hindi dataset**, the project achieved meaningful results in English-to-Hindi translation within limited computational resources.

The inclusion of a **Gradio web interface** makes it simple to interact with the trained model, turning a research experiment into a usable tool.



