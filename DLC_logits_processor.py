from transformers import LogitsProcessor
import torch
import torch.nn.functional as F
from collections import deque

class DLCLogitsProcessor(LogitsProcessor):
    def __init__(self, clip_model, clip_processor, image_features, 
                 tokenizer, window_size=8, penalty_scale=1.0, top_k=50,
                 token_context_weight=0.5, model_type='llava', ispope=False, query=None):  
        super().__init__()
        self.input_context_length = 0
        self.window_size = window_size
        self.penalty_scale = penalty_scale
        self.top_k = top_k
        self.token_context_weight = token_context_weight 
        self.model_type = model_type
        
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.image_features = F.normalize(image_features, p=2, dim=-1)
        self.tokenizer = tokenizer

        self.similarity_buffer = deque(maxlen=8)
        self.buffer_warmup = 3
        self.pope = ispope
        self.query = query

    def _compute_relative_sim(self, current_sim):
        if len(self.similarity_buffer) >= self.buffer_warmup:
            mean = torch.mean(torch.tensor(self.similarity_buffer))
            return mean.item()
        return current_sim
    
    def __call__(self, input_ids, scores):
        current_len = input_ids.shape[1]

        if not self.pope:
            if self.model_type == 'llava':
                if self.input_context_length == 0:
                    self.input_context_length = current_len
                generated_len = current_len - self.input_context_length

                if generated_len < self.window_size:
                    return scores

                start_pos = self.input_context_length + max(0, generated_len - self.window_size)
                recent_ids = input_ids[0, start_pos:current_len]
                recent_text = self.tokenizer.decode(recent_ids, skip_special_tokens=True).strip()
            else:
                if current_len < self.window_size:
                    return scores
                start_pos = max(0, current_len - self.window_size)
                recent_ids = input_ids[0, start_pos:current_len]
                recent_text = self.tokenizer.decode(recent_ids, skip_special_tokens=True).strip()
        else:
            recent_text = self.query
            recent_text = recent_text[0]
        
        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)
        token_strs = [self.tokenizer.decode([tid]) for tid in topk_indices[0]]
        
        valid_tokens = []
        valid_indices = []
        for idx, token in enumerate(token_strs):
            if not token.strip():
                continue
            valid_tokens.append(token)
            valid_indices.append(idx)
            
        if not valid_indices:
            return scores
        
        all_texts = []        
        all_texts.append(recent_text if recent_text else "none")
        context_token_texts = [recent_text + token for token in valid_tokens]
        all_texts.extend(context_token_texts)
        all_texts.extend(valid_tokens)

        with torch.no_grad():
            batch_inputs = self.clip_processor(
                text=all_texts,
                padding=True,
                return_tensors="pt",
                max_length=77,
                truncation=True
            ).to(self.clip_model.device)
            
            all_text_features = self.clip_model.get_text_features(**batch_inputs)
            all_text_features = F.normalize(all_text_features, p=2, dim=-1)

            text_features = all_text_features[0:1]
            context_token_features = all_text_features[1:1+len(valid_tokens)]
            token_features = all_text_features[1+len(valid_tokens):]

            s_t = (self.image_features @ text_features.T).item()
            self.similarity_buffer.append(s_t)

            if len(self.similarity_buffer) <= self.buffer_warmup:
                rel_s_t = s_t
            else:
                rel_s_t = self._compute_relative_sim(s_t)

            base_sim = rel_s_t
            
            sim_context = context_token_features @ self.image_features.T
            sim_context = sim_context.squeeze()
            
            sim_token = token_features @ self.image_features.T
            sim_token = sim_token.squeeze()
        
        w = self.token_context_weight
        sim_v = w * sim_context + (1-w) *sim_token
        
        dynamic_lambda = self.penalty_scale * (1.0 - base_sim)**2
        
        relative_sim = (sim_v - base_sim) / (1 - base_sim + 1e-6)
        penalties = torch.sigmoid(relative_sim)
        
        full_adjusted_scores = topk_scores[0].clone()
        
        for i, (idx, penalty) in enumerate(zip(valid_indices, penalties)):
            full_adjusted_scores[idx] = topk_scores[0][idx] * torch.exp(dynamic_lambda * penalty)
        
        for idx, score in enumerate(full_adjusted_scores):
            scores[0, topk_indices[0, idx]] = score
            
        return scores