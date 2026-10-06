"""The same sequential stream, with explicit cycles and recoverable cursor."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from common import SHORT, CACHE, MICROBATCH, CONTEXT, atomic_json, sha
from data import FineWeb, reference


class Stream(reference.TrainStream):
    def __init__(self, source, batch=MICROBATCH, context=CONTEXT):
        super().__init__(source,batch,context)
        self.cycles=0

    def next_batch(self):
        wrap = (self.shard == len(self.names)-1 and
                self.position+self.batch*self.context+1 > len(self.tokens))
        result=super().next_batch()
        if wrap:
            self.cycles+=1
        return result

    def state_dict(self):
        return dict(shard=self.shard,position=self.position,cycles=self.cycles,
                    batch=self.batch,context=self.context,names=self.names)

    def load_state_dict(self, state):
        if state['names']!=self.names or state['batch']!=self.batch or state['context']!=self.context:
            raise RuntimeError('Resume data ordering/batch mismatch')
        shard,position=int(state['shard']),int(state['position'])
        if not 0 <= shard < len(self.names): raise ValueError('Invalid shard cursor')
        tokens=self.source.array(self.names[shard])
        if not 0 <= position < len(tokens) or position % (self.batch*self.context):
            raise ValueError('Invalid within-shard cursor')
        self.shard,self.position,self.cycles,self.tokens=shard,position,int(state['cycles']),tokens


def corpus_metadata(source):
    files={k:v for k,v in source.manifest['files'].items() if '_train_' in k}
    count=MICROBATCH*CONTEXT
    unique=sum((v['size']-1024)//2 for v in files.values())
    usable=sum((((v['size']-1024)//2-1)//count)*count for v in files.values())
    return dict(repo=source.manifest['repo'],revision=source.manifest['revision'],
                train_shards=len(files),corpus_tokens=unique,usable_tokens_per_epoch=usable,
                epoch_definition='tokens_seen / usable_tokens_per_full_sequential_corpus_pass',
                ordering='unchanged sequential shard traversal; wraps to first shard after a full pass',
                manifest_sha256=sha(SHORT.parent/'speedrun30/data_manifest.json'))


def prepare(root, deadline):
    source=FineWeb(CACHE,deadline)
    names=sorted(source.manifest['files'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name,_ in zip(names,pool.map(source.array,names)):
            print('Verified benchmark shard:',name,flush=True)
    atomic_json(root/'data_receipts.json',source.receipts)
    atomic_json(root/'corpus.json',corpus_metadata(source))
