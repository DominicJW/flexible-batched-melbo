# flexible-batched-melbo
Alternative implementation of z3research/batched_melbo
This has been tested with DominicJW/Ministral-3-3B-Instruct-2512-BF16-bnb-4bit
There is a difference between the vanilla unsteered slice of the steered outputs and the non steered outputs generated from the model with no steering applied
I believe this is due to floating point errors which are amplified by the quantization
Using the code below, generally we get most of the same token ids in the topk, but there can be differences in their order within the topk.
```
probs_v = F.softmax(vanilla_outputs.logits, dim=-1)  
batch_size, seq_len, vocab_size = probs_v.shape
top_k = 10
top_probs_v, top_idx_v = torch.topk(probs_v, k=top_k, dim=-1) 

probs_s = F.softmax(steered_outputs.logits[:num_prompts], dim=-1) 
batch_size, seq_len, vocab_size = probs_s.shape
top_k = 10
top_probs_s, top_idx_s = torch.topk(probs_s, k=top_k, dim=-1)  
```

The purpose of this function is to provide a method which is more intuitive to me for experimenting with steering vectors.
It also allows different source layers, different source token indexes (so only certain positions in the sequences are steered)
and different groupings of prompts to be used for each steering vector. All within the same batch.

However the trade off for this flexibility is ease of use.
