import torch
from transformers import CLIPTokenizer, CLIPTextModel


class TextEncoder:

    def __init__(
        self,
        device
    ):

        self.device = device

        model_name = "openai/clip-vit-base-patch32"

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name
        )

        self.encoder = CLIPTextModel.from_pretrained(
            model_name
        ).to(device)

        self.encoder.eval()

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False


    @torch.no_grad()
    def encode(
        self,
        prompts
    ):

        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        tokens = {
            key: value.to(self.device)
            for key, value in tokens.items()
        }

        output = self.encoder(
            **tokens
        )

        return output.last_hidden_state