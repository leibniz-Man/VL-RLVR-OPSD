import json, math, sys
from pathlib import Path
import torch
from datasets import load_dataset
from jinja2 import Template
sys.path.insert(0,'/home/coder/lhc/CEPO/custom_experiments/cepo_imgmask_teacher')
from tools.entropy_probe import prepare_inputs, append_response, first_image, mask_image
from transformers import AutoModelForImageTextToText, AutoProcessor
model_path='/home/coder/lhc/CEPO/custom_experiments/cepo_imgmask_teacher/checkpoints/cepo/qwen3_vl_2b_geo_cepo_imgmask_teacher/global_step_50/actor/huggingface'
result_path=Path('/home/coder/lhc/CEPO/custom_experiments/cepo_imgmask_teacher/logs/entropy_probe_visual_error_sample32.json')
d=json.loads(result_path.read_text()); records=d['all_token_records']; ids=torch.tensor([r['token_id'] for r in records],dtype=torch.long)
proc=AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
model=AutoModelForImageTextToText.from_pretrained(model_path,dtype=torch.bfloat16,device_map={'':0},trust_remote_code=True).eval()
ex=load_dataset('hiyouga/geometry3k',split='test')[32]
tmpl=Template(Path('/home/coder/lhc/CEPO/examples/format_prompt/math_short.jinja').read_text())
rendered=tmpl.render(content=ex['problem']); img=first_image(ex); mask=mask_image(img)
pv=prepare_inputs(proc,rendered,img,torch.device('cuda')); pm=prepare_inputs(proc,rendered,mask,torch.device('cuda'))
plen=pv['input_ids'].shape[1]
with torch.inference_mode():
 ov=model(**append_response(pv,ids,torch.device('cuda'))).logits[0]
 om=model(**append_response(pm,ids,torch.device('cuda'))).logits[0]
for t in [15,18,19,23,26,27]:
 print('\nTOKEN',t,repr(records[t]['decoded_token']),'G',records[t]['gap'],'Hvis',records[t]['h_vis'],'Hmask',records[t]['h_mask'])
 for name,logits in [('VIS',ov[plen-1+t]),('MASK',om[plen-1+t])]:
  lp=torch.log_softmax(logits.float(),dim=-1); vals,idx=torch.topk(lp,10)
  print(name)
  for v,i in zip(vals.tolist(),idx.tolist()): print(repr(proc.tokenizer.decode([i],skip_special_tokens=False)),f'p={math.exp(v):.6f}',f'logp={v:.4f}')
