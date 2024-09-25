import contextlib
from model.med import BertConfig, BertModel
from transformers import BertTokenizer, LlamaForCausalLM, LlamaTokenizer#, BertLMHeadModel

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

import copy

#from model.attention import Transformer3DModel
from model.blip import create_vit, init_tokenizer, load_checkpoint
from model.blip_pretrain import BLIP_Pretrain
from model.swin import swin_3d_tiny, SwinTransformer3D, SwinTransformer2D
from model.Qformer import BertLMHeadModel


from torch.nn import TransformerDecoderLayer, TransformerDecoder
from timm.models.vision_transformer import vit_base_patch16_224



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class T2VQA(nn.Module):
    def __init__(self,
                 args):
        super().__init__()

        med_config = args['med_config']
        image_size = args['image_size']
        embed_dim = args['embed_dim']
        llm_model = args['llm_model']

        self.blip = BLIP_Pretrain(image_size = image_size, vit = 'large', embed_dim = embed_dim, med_config = med_config)
        state_dict = torch.load(args['blip_weights'], map_location='cpu')
        self.blip.load_state_dict(state_dict["model"], strict=False)

        for name, param in self.blip.named_parameters():
            if ("text_encoder" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.finetune_text_proj = nn.Linear(self.blip.text_encoder.config.hidden_size, embed_dim)

        encoder_config = BertConfig.from_pretrained(args['bert_weights'])
        encoder_config.encoder_width = embed_dim
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = 32

        self.finetune_Qformer = BertLMHeadModel.from_pretrained(
            args['bert_weights'], config=encoder_config
        )



        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False)
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
        )

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        self.finetune_llm_proj = nn.Linear(embed_dim, self.llm_model.config.hidden_size)

        self.finetune_proj = nn.Linear(
            self.finetune_Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )
        

        for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        self.llm_model = self.llm_model.eval()
        self.llm_model.train = disabled_train

        self.excellent_idx, self.good_idx, self.fair_idx, self.poor_idx, self.bad_idx = self.llm_tokenizer(["excellent", "good","fair", "poor", "bad"])['input_ids']
        self.excellent_idx = self.excellent_idx[1]
        self.good_idx = self.good_idx[1]
        self.fair_idx = self.fair_idx[1]
        self.poor_idx = self.poor_idx[1]
        self.bad_idx = self.bad_idx[1]

        self.swin3d = swin_3d_tiny()
        state_dict = torch.load(args['swin_weights'], map_location='cpu')
        state_dict = state_dict['state_dict']
        
        i_state_dict = OrderedDict()
        for key in state_dict.keys():
            if "head" in key:
                continue
            if "cls" in key:
                tkey = key.replace("cls", "vqa")
            elif "backbone" in key:
                tkey = key.replace("backbone.", "")
                i_state_dict[tkey] = state_dict[key]
            else:
                i_state_dict[key] = state_dict[key]
            
        print(self.swin3d.load_state_dict(i_state_dict, strict=False))
        
        self.swin_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.weights = torch.Tensor([[1], [2], [3], [4], [5]])

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block


    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, data, caption, prompt):

        video = data['video']

        f = self.swin3d(video)
        f = self.swin_avg_pool(f)
        f = f.view(f.size(0), -1)
        f = f.unsqueeze(1)
        inputs_swin = f.expand(-1, 32, -1).to(video.device)
        
        atts_swin = torch.ones(inputs_swin.size()[:-1], dtype=torch.long).to(video.device)

        inputs_llm = []

        text = self.blip.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                                  return_tensors="pt").to(video.device)

        img_feats = []
        
        for j in range(video.size(2)):
            image = video[:,:,j,:,:]

            image_embeds = self.blip.visual_encoder(image)


            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(video.device)
            output = self.blip.text_encoder(text.input_ids,
                                                attention_mask = text.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,
                                                return_dict = True,
                                            )

            output = self.finetune_text_proj(output.last_hidden_state[:,0,:])


            inputs_llm.append(output)
            img_feats.append(image_embeds)

        img_feats = torch.stack(img_feats, dim=1)
        image_atts = torch.ones(img_feats.size()[:-1],dtype=torch.long).to(video.device)

        inputs_llm = torch.stack(inputs_llm, dim=1)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(video.device)
        
        all_inputs = self.finetune_Qformer.bert(
                query_embeds=inputs_swin,
                attention_mask=atts_swin,
                encoder_hidden_states=inputs_llm,
                encoder_attention_mask=atts_llm,
                return_dict=True,
            )

        inputs_llm = self.finetune_proj(all_inputs.last_hidden_state[:,:inputs_swin.size(1),:])
   

        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(video.device)

        

        llm_tokens = self.llm_tokenizer(
            [prompt] * video.size(0),
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        
        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)

            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,

                )

        output_logits = outputs.logits[:, -1]

        lexcellent, lgood, lfair, lpoor, lbad = output_logits[:, self.excellent_idx], output_logits[:, self.good_idx], output_logits[:, self.fair_idx], output_logits[:,self.poor_idx], output_logits[:, self.bad_idx]
        q_pred = (torch.stack([lexcellent, lgood, lfair, lpoor, lbad]) / 100).softmax(0)

        weights = self.weights.expand(-1, q_pred.shape[1]).to(video.device)
        q_pred = torch.mul(q_pred, weights)

        q_pred = torch.sum(q_pred, dim=0)

        return q_pred







if __name__=="__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model = T2VQA(med_config='../configs/med_config.json', image_size = 224).to(device)
    model.eval()
    caption = 'A random caption'
    prompt = 'Please assess the quality of this image'
    video = torch.randn(2, 3, 8, 224, 224).to(device)

    with torch.no_grad():
        output = model(video, caption, prompt)
    print(output)        


