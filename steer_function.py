import torch
def make_hook_fn(steering_vectors,prompt_indexes,token_indexes):
    def f(m,i,o):
        #steering_vectors is of shape num_vec,hidden_size
        
        #steering_tensor_tok_indexed is shape num_vec_a,num_prompt,max_seq_len,hidden_size
        num_vecs = steering_vectors.shape[0]
        hidden_size = steering_vectors.shape[1]
        num_prompts = prompt_indexes.shape[1]
        max_seq_len = token_indexes.shape[2]

        #asume everything except o is on cpu
        steering_tensor_tok_indexed = steering_vectors.unsqueeze(1).unsqueeze(1) + torch.zeros((num_vecs,num_prompts,max_seq_len,hidden_size))
    
        #apply the token mask (to only steer the tokens we are interested in) token mask is shape num_vec,num_prompt,max_seq_len
        #steering_tensor_tok_indexed[i,j,k,:] Some of these will then be zeroed (the output is still needed, it must be unsteered though)
        steering_tensor_tok_indexed[~token_indexes] = 0
        #must do this first, as must not over-write the nan        
        
        #apply the prompt mask (to only steer the prompts we are interested in) prompt mask is shape num_vec,num_prompt
        #steering_tensor_tok_indexed[i,j,:,:] some of these are naned as the output is not needed and are later indexed away
        steering_tensor_tok_indexed[~prompt_indexes] = torch.nan
        
        assert len(o.shape) == 3
        assert steering_tensor_tok_indexed.shape[2] == o.shape[1] or steering_tensor_tok_indexed.shape[2] == 1 #usage assertion
        
        vanilla = o[:num_prompts]

        temp = vanilla+ torch.zeros((num_vecs,)+vanilla.shape).to(device = o.device,dtype = o.dtype)
        final_output = temp + steering_tensor_tok_indexed.to(device = o.device,dtype = o.dtype)
        
        #development assetion checking
        assert torch.isnan(final_output[~prompt_indexes]).all().item()

        if max_seq_len ==1 :
            assert (temp[prompt_indexes][~token_indexes[prompt_indexes].squeeze(-1)] == final_output[prompt_indexes][~token_indexes[prompt_indexes].squeeze(-1)]).all().item()
        else:
            assert (temp[prompt_indexes][~token_indexes[prompt_indexes]] == final_output[prompt_indexes][~token_indexes[prompt_indexes]]).all().item() #wierd indexing is to get rid of the nans!        
        
        final_output = final_output.reshape((final_output.shape[0]*final_output.shape[1],)+final_output.shape[2:])#flatten to num_vec*num_prompts,max_seq_len,hs
        prompt_indexes_flat_view = prompt_indexes.reshape(prompt_indexes.shape[0]*prompt_indexes.shape[1]).to(device=o.device) #must be bool type
        final_output = final_output[prompt_indexes_flat_view]#remove the nan slices

        final_output = torch.concat((o,final_output),dim=0)
        print(final_output.shape)
        return final_output
        
    return f

from contextlib import contextmanager
@contextmanager
def steer(model_layers,source_layers,steering_vectors_all_layers,prompt_indexes_all_layers,token_indexes_all_layers):
'''
    source_layers [int] len:num_source_layers
    steering_vectors_all_layers [shape:(num_vecs_in_layer,hs)] len:num_source_layers (dtype = float like)
    prompt_indexes_all_layers [shape: (num_vecs_in_layer,num_prompts)] len:num_source_layers (dtype = bool like)
    token_indexes_all_layers [shape: (num_vecs_in_layer, num_prompts,max_seq_len)] len:num_source_layers (dtype = bool like)

  usage:
      input_text = ["Your input text goes here.","Your input text goes here.","Your input text goes here.","Your input text goes here."]
      inputs = tokenizer(input_text, return_tensors="pt")      

      hidden_size = model.config.hidden_size
      num_prompts = inputs["input_ids"].shape[0]
      
      source_layers_ = [1,4,20]
      num_vecs_in_layer = [4,7,8]
      
      steering_vectors_all_layers_ = [torch.rand((num_vecs_in_layer[i],hidden_size)) for i in range(len(source_layers_))]      
      prompt_indexes_all_layers_ = [torch.rand((vec_tensor.shape[0],num_prompts))>0.0 for vec_tensor in steering_vectors_all_layers_]
      
      max_seq_len = inputs["input_ids"].shape[1]
      # max_seq_len = 1 #to steer all tokens
      
      token_indexes_all_layers_ = [torch.rand((vec_tensor.shape[0],num_prompts,max_seq_len))>0.0 for vec_tensor in steering_vectors_all_layers_] #>0.0 means all tokens
    
      with steer(model_layers,source_layers_,steering_vectors_all_layers_,prompt_indexes_all_layers_,token_indexes_all_layers_):
          steered_outputs = model(**inputs)
'''
    assert len(source_layers) == len(token_indexes_all_layers) == len(prompt_indexes_all_layers) == len(steering_vectors_all_layers) #check no extra or missing layers
    assert all([prompt_indexes_all_layers[i].shape[0] == steering_vectors_all_layers[i].shape[0] for i in range(len(source_layers))]) #check number of vectors correct
    assert all([token_indexes_all_layers[i].shape[0:2] == prompt_indexes_all_layers[i].shape[0:2] for i in range(len(source_layers))]) #
    assert all([prompt_indexes_all_layers[0].shape[1] == prompt_indexes_all_layers[i].shape[1] for i in range(len(source_layers))]) #same num_prompts across layers
    assert all([token_indexes_all_layers[0].shape[2] == token_indexes_all_layers[i].shape[2]  for i in range(len(source_layers))]) #same max_seq_len across layers
    
    
    model_layers = model.model.language_model.layers
    handles = [0 for _ in source_layers]
    try:
        for i,layer_idx in enumerate(source_layers):
            handles[i] = model_layers[layer_idx].register_forward_hook(
                make_hook_fn(
                    steering_vectors_all_layers[i],
                    prompt_indexes_all_layers[i],
                    token_indexes_all_layers[i]
                )
            )
        yield
    finally:
        for handle in handles:
            handle.remove()
