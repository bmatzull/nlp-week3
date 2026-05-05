from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Generator:
    def __init__(self, model_name: str):
        """Initializes the text generation pipeline"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate(self, question: str, contexts: list[str], k: int) -> str:
        """Prompts the model with the retrieved context to answer the question"""
        if k == 0 or not contexts:
            prompt = f"Answer the following question.\nQuestion:{question}\nAnswer:"
        else:
            combined_context = "\n".join(contexts)
            prompt = f"Answer the following question based on the context.\nContext:{combined_context}\nQuestion: {question}\nAnswer:"

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=50)

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"Error: {str(e)}"