import os
from tqdm import tqdm
import gc
import torch

from egh_vlm.hallu_dataset import HalluDataset, save_hallu_dataset, load_hallu_dataset
from egh_vlm.utils import Qwen3ModelBundle


def extract_features_qwen3(
    context_messages: list, 
    answer_messages: list, 
    model_bundle: Qwen3ModelBundle, 
    targeted_layer: int = -1):
    """
    Return embed & grad features
    """
    model, processor, device = model_bundle.model, model_bundle.processor, model_bundle.device
    
    c_ids = processor.apply_chat_template(
        context_messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors='pt'
    )
    a_ids = processor.apply_chat_template(
        answer_messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors='pt'
    )
    # Move tensors to device
    c_ids = {k: v.to(device) for k, v in c_ids.items()}
    a_ids = {k: v.to(device) for k, v in a_ids.items()}

    with torch.set_grad_enabled(True):
        model.eval()

        c_output = model(**c_ids, output_hidden_states=True)
        a_output = model(**a_ids, output_hidden_states=True)
        c_length = c_ids['input_ids'].shape[1]
        a_length = a_ids['input_ids'].shape[1]

        c_prob = c_output.logits.squeeze(0)[c_length - a_length + 1:, :]
        a_prob = a_output.logits.squeeze(0)[1:, :]

        c_vector = c_output.hidden_states[targeted_layer]
        a_vector = a_output.hidden_states[targeted_layer]

        # Compute KL divergence & gradient
        c_prob_float = c_prob.float()
        a_prob_float = a_prob.float()
        a_prob_softmax = a_prob_float.softmax(dim=-1)
        kl_divergence = torch.sum(
            a_prob_softmax * (a_prob_softmax.log() - torch.log_softmax(c_prob_float, dim=-1))
        )
        try:
            grad = torch.autograd.grad(
                outputs=kl_divergence, inputs=a_vector, create_graph=False, allow_unused=True,
            )[0]
            if grad is None:
                raise RuntimeError('Gradient is None.')
            grad = grad.squeeze(0)[1:, :]
        except Exception:
            # Fallback to zeros if gradient computation fails
            grad = torch.zeros_like(a_vector.squeeze(0)[1:, :])

        # Compute embedding difference
        a_emb = a_vector.squeeze(0)[1:, :]
        c_emb = c_vector.squeeze(0)[c_length - a_length + 1:, :]
        emb = c_emb - a_emb

    emb_cpu = emb.detach().to('cpu')
    grad_cpu = grad.detach().to('cpu')

    # Clean up resources
    del c_ids, a_ids
    del c_output, a_output
    del c_prob, a_prob, c_prob_float, a_prob_float, a_prob_softmax, kl_divergence
    del c_vector, a_vector, a_emb, c_emb, emb, grad

    return {
        'emb': emb_cpu,
        'grad': grad_cpu
    }

def extract_features(
    dataset: any,
    model_bundle: any,
    client_type: str='qwen3',
    save_path: str=None,
    save_interval: int=20,
    mask_mode: str=None,
    targeted_layer: int=-1):
    '''
    model_bundle: Qwen3ModelBundle
    client_type: 'qwen3'
    mask_mode: None, 'image' or 'question'
    targeted_layer: The layer index from which to extract features
    '''
    
    if client_type not in ['qwen3']:
        print('Unsupported client')
        return None
    
    if mask_mode not in [None, 'image', 'question']:
        print('Incorrect mask mode')
        return None
    
    # Load processed_features
    processed_features = HalluDataset()
    
    if save_path is not None and os.path.exists(save_path):
        processed_features = load_hallu_dataset(save_path)
    processed_ids = set(processed_features.ids)
    
    if client_type == 'qwen3':
        for item in tqdm(dataset, desc=f'Extracting features for client {client_type}'):
            if item['id'] in processed_ids:
                continue
            
            question = item['question']
            image_path = item['image_path']
            answer = item['answer']
            
            # Construct messages context
            context = []
        
            if image_path is not None and mask_mode != 'image':
                context.append({'type': 'image', 'image': image_path})
            if question is not None and mask_mode != 'question':
                context.append({'type': 'text', 'text': question})

            # Construct messages
            context_messages = [
                {'role': 'user', 'content': context},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}
            ]
            answer_messages = [
                {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}
            ]
            
            # Extract features
            features = extract_features_qwen3(
                context_messages=context_messages,
                answer_messages=answer_messages,
                model_bundle=model_bundle,
                targeted_layer=targeted_layer
            )
            emb = features['emb']
            grad = features['grad']
            
            # Exclude empty, NaN, and inf features
            has_non_empty_features = emb.numel() > 0 and grad.numel() > 0
            has_valid_values = (
                not torch.isnan(emb).any()
                and not torch.isinf(emb).any()
                and not torch.isnan(grad).any()
                and not torch.isinf(grad).any()
            )
            
            if has_non_empty_features and has_valid_values:
                processed_features.add_item(item['id'], emb, grad, item['label'])
            else:
                print(f"Skipping id={item['id']} due to invalid features.")
                continue
            
            # Save features
            if save_path is not None and len(processed_features) % save_interval == 0:
                save_hallu_dataset(processed_features, save_path)

            # Clean up resources
            del features, emb, grad, context_messages, answer_messages, context
            # Clean up GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Final save
        if save_path is not None:
            save_hallu_dataset(processed_features, save_path)
            
        return processed_features
    else:
        print('Unsupported client')
        return None
