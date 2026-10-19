import torch, time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
M="facebook/nllb-200-3.3B"
tok=AutoTokenizer.from_pretrained(M, src_lang="eng_Latn")
m=AutoModelForSeq2SeqLM.from_pretrained(M, torch_dtype=torch.float16).cuda().eval()
W=["Greek","Sweden","Iran","Iceland","French","Pali","Granada","Lima","Seven","Caesar","Ala","Jordan","Hogwarts","Gryffindor","Butterbeer"]
def tr(xs, lang, beams=4, **kw):
    enc=tok(xs,return_tensors="pt",padding=True).to("cuda")
    out=m.generate(**enc,num_beams=beams,max_new_tokens=64,forced_bos_token_id=tok.convert_tokens_to_ids(lang),**kw)
    return tok.batch_decode(out,skip_special_tokens=True)
for lang in ["ita_Latn","tha_Thai"]:
    print(lang)
    a=tr(W,lang); b=tr([w+"." for w in W],lang); c=tr([f"Answer: {w}" for w in W],lang)
    for x in zip(W,a,b,c): print("  ",x)
xs=["Which ancient Greek philosopher is universally recognized as a foundational figure of Western philosophy?"]*64
for flag in [True, False]:
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=flag
    torch.cuda.synchronize(); t=time.time(); tr(xs,"ita_Latn"); torch.cuda.synchronize(); print("flag",flag,"64 long beam4",time.time()-t, flush=True)
    torch.cuda.synchronize(); t=time.time(); tr(W*4,"ita_Latn"); torch.cuda.synchronize(); print("flag",flag,"60 short beam4",time.time()-t, flush=True)
t=time.time(); xs=["Which ancient Greek philosopher is universally recognized as a foundational figure of Western philosophy?"]*64
tr(xs,"ita_Latn"); print("64 long, beam4", time.time()-t)
t=time.time(); tr(xs,"ita_Latn",beams=2); print("64 long, beam2", time.time()-t)
t=time.time(); tr(xs[:32],"ita_Latn"); print("32 long, beam4", time.time()-t)
