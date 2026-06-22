#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
from typing import Any, Dict, List
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

def messages(prompt: str):
    content=[]
    for i, part in enumerate(prompt.split('<image>')):
        if i: content.append({'type':'image'})
        if part: content.append({'type':'text','text':part})
    if not any(x.get('type')=='image' for x in content): content.insert(0,{'type':'image'})
    return [{'role':'user','content':content}]

def extract_answer(text: str):
    boxes=re.findall(r'\\boxed\{([^{}]*)\}', text)
    raw=boxes[-1] if boxes else ''
    if not raw:
        nums=re.findall(r'-?\d+(?:\.\d+)?', text)
        raw=nums[-1] if nums else ''
    return raw.strip()

def numeric(text: str):
    s=text.replace(',','').replace('^\\circ','').replace('°','').replace('$','')
    m=re.search(r'-?\d+(?:\.\d+)?',s)
    return float(m.group()) if m else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True);ap.add_argument('--prompt-template',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--start-index',type=int,default=0)
    ap.add_argument('--num-samples',type=int,default=40);ap.add_argument('--max-new-tokens',type=int,default=512)
    args=ap.parse_args();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    proc=AutoProcessor.from_pretrained(args.model,trust_remote_code=True)
    model=AutoModelForImageTextToText.from_pretrained(args.model,dtype=torch.bfloat16,device_map={'':0},trust_remote_code=True).eval()
    ds=load_dataset('hiyouga/geometry3k',split='test');tmpl=Template(Path(args.prompt_template).read_text())
    rows=[]
    with torch.inference_mode():
        for n,idx in enumerate(range(args.start_index,min(len(ds),args.start_index+args.num_samples)),1):
            ex=ds[idx];img=ex['images'][0].convert('RGB');rendered=tmpl.render(content=ex['problem'])
            chat=proc.apply_chat_template(messages(rendered),add_generation_prompt=True,tokenize=False)
            inp=proc(images=[img],text=[chat],add_special_tokens=False,return_tensors='pt')
            inp={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in inp.items()};plen=inp['input_ids'].shape[1]
            gen=model.generate(**inp,max_new_tokens=args.max_new_tokens,do_sample=False)
            ids=gen[0,plen:];resp=proc.tokenizer.decode(ids.tolist(),skip_special_tokens=True)
            pred=extract_answer(resp);pv=numeric(pred);gv=numeric(str(ex['answer']))
            correct=(pv is not None and gv is not None and abs(pv-gv)<1e-6)
            row={'sample_index':idx,'problem':ex['problem'],'ground_truth':ex['answer'],'predicted_answer':pred,'predicted_numeric':pv,'ground_truth_numeric':gv,'correct_numeric':correct,'num_tokens':int(ids.numel()),'response':resp}
            rows.append(row);out.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
            print(f'[{n}/{args.num_samples}] idx={idx} gt={ex["answer"]!r} pred={pred!r} correct={correct} tokens={ids.numel()}',flush=True)
    print('WROTE',out,flush=True)
if __name__=='__main__':main()
