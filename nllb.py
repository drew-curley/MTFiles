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

translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang="eng_Latn", tgt_lang='npi_Deva', max_length = 1000000)
 
x = """
The gospel of John has 21 chapters. This plan will help you read all of them in 24 days. Each day, read the Scripture and observe what it says. Answer the questions. Then think about the passage, what you have learned, and how you can apply it to your life. Use the checkboxes to keep track of your progress. 
Day 1
☐  Read John 1:1-5
•	What do you learn about God? 
☐  Read John 1:6-13
•	What does a witness do? 
•	How can you be a good witness?
•	Describe the true Light.



”"""


for line in x.splitlines():
    print(translator(line)[0]["translation_text"])
               
