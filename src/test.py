import sys
import os
RWKV_PATH ='/home/rwkv/Peter/RWKV_LM_EXT-main'
sys.path.append(RWKV_PATH)
from helpers import start_proxy, ServiceWorker
from src.model_run import RWKV,create_empty_args,load_embedding_ckpt_and_parse_args,BiCrossFusionEncoder,generate,enable_lora
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
import sqlitedict
import torch

class LLMService:
    def __init__(self,base_model_file,be_lora_file,ce_lora_file,tokenizer_file,device="cuda:0") -> None:
        args = create_empty_args()
        w = load_embedding_ckpt_and_parse_args(base_model_file, args)
        model = RWKV(args)
        info = model.load_state_dict(w)
        print(f'load info {info}')
        self.device = device
        tokenizer = TRIE_TOKENIZER(tokenizer_file)
        self.tokenizer = tokenizer
        dtype = torch.bfloat16
        self.dtype = dtype
        self.model = model.to(device=device,dtype=dtype)

        self.fusedEncoder = BiCrossFusionEncoder(model,be_lora_file,ce_lora_file,tokenizer,device=device,dtype=dtype,lora_type='lora',lora_r=8,lora_alpha=32,add_mlp=True,mlp_dim=1024,target_modules=['emb','ffn.key','ffn.value','ffn.receptance'],cross_adapter_name='cross_encoder_lora',original_cross_adapter_name='embedding_lora',bi_adapter_name='bi_embedding_lora',original_bi_adapter_name='embedding_lora',sep_token_id = 2)
        from rwkv.utils import PIPELINE_ARGS
        self.gen_args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.8, top_k = 100, # top_k = 0 then ignore
                        alpha_frequency = 0.25,
                        alpha_presence = 0.25,
                        alpha_decay = 0.996, # gradually decay the penalty
                        token_ban = [], # ban the generation of some tokens
                        token_stop = [0,2], # stop generation whenever you see any token here
                        chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)
    def get_embeddings(self,inputs):
        with torch.no_grad():
            if isinstance(inputs,str):
                inputs = [inputs]
            outputs = [self.fusedEncoder.encode_texts(input).tolist() for input in inputs]
            return outputs
        
    def get_cross_scores(self,texts_0,texts_1):
        with torch.no_grad():
            assert isinstance(texts_0,list) and isinstance(texts_1,list)
            outputs = [self.fusedEncoder.cross_encode_texts(text_0,text_1).item() for text_0,text_1 in zip(texts_0,texts_1)]
            return outputs
        
    def generate_texts(self, ctx, token_count=100):
        enable_lora(self.model,enable=False)
        with torch.no_grad():
            with torch.autocast(enabled=True,device_type='cuda',dtype=self.dtype):
                out_str = generate(self.model, ctx,self.tokenizer,token_count=token_count,args=self.gen_args,device=self.device)
        enable_lora(self.model,enable=True)
        return out_str  

class ServiceWorker(ServiceWorker):
    def init_with_config(self, config):
        base_model_file = config["base_model_file"]
        bi_lora_path = config["bi_lora_path"]
        cross_lora_path = config["cross_lora_path"]
        chat_lora_path = config["chat_lora_path"]  
        tokenizer_file = config["tokenizer_file"]
        device = config["device"]  
        self.llm_service = LLMService(base_model_file, bi_lora_path, cross_lora_path, chat_lora_path, tokenizer_file, device)
    
    
    def process(self, cmd):
        if cmd['cmd'] == 'GET_EMBEDDINGS':
            texts = cmd["texts"]
            value = self.llm_service.get_embeddings(texts)
            return value
        elif cmd['cmd'] == 'GET_CROSS_SCORES':
            texts_0 = cmd["texts_0"]
            texts_1 = cmd["texts_1"]
            value = self.llm_service.get_cross_scores(texts_0,texts_1)
            return value
        elif cmd['cmd'] == 'BEAM_GENERATE':
            instruction = cmd["instruction"]
            input_text = cmd["input_text"]
            token_count = cmd.get("token_count", 128)
            num_beams = cmd.get("num_beams", 5)
            value=self.llm_service.beam_generate(instruction, input_text, token_count, num_beams)
            return value
        elif cmd['cmd'] == 'SAMPLING_GENERATE':
            instruction = cmd["instruction"]
            input_text = cmd["input_text"]
            token_count = cmd.get("token_count", 128)
            temperature = cmd.get("temperature", 1.0)
            top_p = cmd.get("top_p", 0)
            value=self.llm_service.sampling_generate(instruction, input_text, token_count, temperature, top_p)     
            return value       
        return ServiceWorker.UNSUPPORTED_COMMAND


if __name__ == '__main__':
    base_rwkv_model = '/home/rwkv/Peter/model/base/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
    bi_lora_path = '/home/rwkv/Peter/model/bi/RWKV-x060-World-1B6-v2_rwkv_lora.pth'
    cross_lora_path = '/home/rwkv/Peter/model/cross/RWKV-x060-World-1B6-v2.1-20240328-ctx4096-5000k.pth'
    tokenizer_file = os.path.join('/home/rwkv/Peter/RWKV_LM_EXT-main/tokenizer/rwkv_vocab_v20230424.txt')
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    chat_lora_path = '/home/rwkv/Peter/model/chat/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
    chat_pissa_path = '/home/rwkv/Peter/model/chat/init_pissa.pth'
    chat_lora_r = 64
    chat_lora_alpha = 64
    LLMService(
        base_rwkv_model,
        bi_lora_path,
        cross_lora_path,
        chat_lora_path,
        tokenizer,
        chat_lora_r=chat_lora_r,
        chat_lora_alpha=chat_lora_alpha,
        chat_pissa_path=chat_pissa_path)

    texts = ['我打算取消订单','我要取消订单','我要退货','我要退款']
    outputs = LLMService.encode_texts(texts)
    outputs = torch.tensor(outputs)
    print(outputs)
    from sentence_transformers.util import pairwise_cos_sim
    for qid in range(len(texts)):
        query = outputs[qid]
        for i in range(len(texts)):
            if i != qid:
                print(f'{texts[qid]} vs {texts[i]} is {pairwise_cos_sim(query.unsqueeze(0),outputs[i].unsqueeze(0))}')

        print('-----------------------')
    

    texts_a = ['FAQ是什么？','FAQ是什么？','FAQ是什么？','FAQ是什么？']
    texts_b = ['下图是百度百科对FAQ的解释，我们可以简单的理解其为，网站中的常见问题帮助中心。采用一问一答的方式帮助客户快速解决产品/服务问题！','：FAQ(Frequently Asked Questions)问答系统是目前应用最广泛的问答系统。这种问答系统的结构框架明了、实现简单、容易理解，非常适合作为问答系统入门学习时的观察对象。这里基于本人在问答系统建设方面的“多年”经验，对FAQ问答相关的定义、系统结构、数据集建设、关键技术、应用等方面进行了整理和介绍。','FAQ是英文 Frequently Asked Questions的缩写。中文意思是“常见问题”，或者更通俗点说，“常见问题解答”。FAQ是目前互联网上提供在线帮助的主要方式，通过事先组织一些常见的问答，在网页上发布咨询服务。','从技术，即实现方式的角度来看，问答系统有很多种，包括基于FAQ的问答、基于知识图谱的问答、基于文本的问答等等。这里围绕应用最为广泛的FAQ问答系统，对问答系统的定义、思想、基本结构、方法和应用价值进行介绍。']
    outputs = LLMService.cross_encode_texts(texts_a,texts_b)
    print(outputs)


    instruction ='根据给定的短文，回答以下问题：黄循财的是哪国人？'
    input_text = '黄循财（英语：Lawrence Wong Shyun Tsai，1972年12月18日—），新加坡华裔政治人物，现任新加坡总理兼财政部部长、人民行动党社区基金会主席。他与王乙康和颜金勇共同主持了因应新加坡2019冠状病毒病大流行的多部委工作组。曾任新加坡副总理，教育部、国家发展部、文化、社区及青年部的部长，通讯及新闻部和财政部的第二部长，以及人民行动党副秘书长。[1]黄循财是人民行动党第四代领导层，也是人民行动党中央执行委员会首任副秘书长兼政策论坛顾问。'
    output = LLMService.sampling_generate(instruction,input_text)
    print(output)

    beam_results = LLMService.beam_generate(instruction,input_text)
    for result in beam_results:
        print(result)