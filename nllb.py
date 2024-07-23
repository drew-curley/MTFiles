# run off of GPU in PyTorch
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# check if cuda is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available, using CPU instead.")

######


# repo = "facebook/nllb-200-3.3B"
repo = "facebook/nllb-200-3.3B"


model = AutoModelForSeq2SeqLM.from_pretrained(repo)
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(repo)

translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang="eng_Latn", tgt_lang='afr_Latn', max_length = 1000000)
 
x = """
Susan is the greatest worker. 



”"""


for line in x.splitlines():
    print(translator(line)[0]["translation_text"])
               
