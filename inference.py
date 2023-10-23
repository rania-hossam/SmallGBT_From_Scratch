from dataclasses import dataclass
from gpt import GPT
from transformers import GPT2TokenizerFast
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# modify the parameters here
max_length = 512
model_path = "models/smallGPT.pth"
tokenizer_path = "tokenizer/tokenizer16384_v2.json"
n_tokens = 1000
temperature = 0.8
top_k = 0
top_p = 0.9

tokenizer = GPT2TokenizerFast(tokenizer_file=tokenizer_path)

@dataclass
class GPTConfig:
    n_embd = 768
    vocab_size = len(tokenizer.get_vocab())
    max_length = 512
    n_head = 8
    n_layer = 8
    dropout = 0.0
    training = True
    pad_token = tokenizer.convert_tokens_to_ids('[PAD]')
    
config = GPTConfig
model = GPT(config)

model_stat = torch.load(model_path)
model.load_state_dict(model_stat["model_state_dict"])
model = model.to(device)

context = '''
Alex: Good morning, everyone. I'm thrilled to be here today to discuss our team's outstanding performance over the last quarter. We have some great news to share, and we also have a special guest with us. Let me introduce you to Sarah Smith!

Sarah: Thank you, Alex. I'm honored to join the team as the new Head of Marketing. I've just moved here from the West Coast, so adapting to the fast pace of this city has been a delightful challenge, but I'm loving every moment of it.

Alex: We're delighted to have you, Sarah. Your fresh perspective and experience are already making a positive impact on our team. Before we dive into the details, let's hear from our Finance Director, John Adams, on how we've performed financially this quarter.

John: Thank you, Alex. I'm pleased to report that our financial results this quarter are exceptional. We've not only met but exceeded our revenue targets, and our cost-saving initiatives have shown remarkable progress. I want to acknowledge the hard work and dedication of our finance team for making this happen.

Alex and Sarah: Well done, team!

'''
context = torch.tensor(tokenizer.encode(context), dtype=torch.long, device=device).reshape(1, -1).to(device)
print(
    tokenizer.decode(
        model.generate(
            context, max_tokens_generate=n_tokens, top_k=top_k, top_p=top_p, temperature=temperature
        ).tolist()
    )
)