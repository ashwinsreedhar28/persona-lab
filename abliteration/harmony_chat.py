"""Display only the assistant's final Harmony channel; never execute tool messages."""


class FinalChannel:
    def __init__(self,tokenizer):
        self.tok=tokenizer
        keys=('im_start','im_sep','im_end','meta_sep','fim_suffix','ghissue')
        schemes=[('start','message','end','channel','return','call'),
                 ('im_start','im_sep','im_end','meta_sep','fim_suffix','ghissue')]
        self.ids={}
        for names in schemes:
            ids={key:tokenizer.convert_tokens_to_ids('<|'+name+'|>') for key,name in zip(keys,names)}
            if all(v is not None and v!=tokenizer.unk_token_id for v in ids.values()) and len(set(ids.values()))==len(keys):
                self.ids=ids;self.channel_marker=tokenizer.decode([ids['meta_sep']],skip_special_tokens=False);break
        if not self.ids:
            raise ValueError('Tokenizer lacks distinct required Harmony markers')
        self.header=[];self.in_header=True;self.visible=False;self.tokens=[]

    def put(self,token):
        if token==self.ids['im_start']:
            self.header=[];self.in_header=True;self.visible=False;return False
        if token in (self.ids['im_end'],self.ids['fim_suffix'],self.ids['ghissue']):
            self.visible=False;self.in_header=True;self.header=[];return False
        if self.in_header:
            if token==self.ids['im_sep']:
                text=self.tok.decode(self.header,skip_special_tokens=False)
                parts=text.split(self.channel_marker)
                sender=parts[0].strip()
                self.visible=(len(parts)==2 and parts[1].strip()=='final' and sender in ('','assistant'))
                self.in_header=False
            else:self.header.append(token)
            return False
        if self.visible:
            self.tokens.append(token);return True
        return False

    def text(self):return self.tok.decode(self.tokens,skip_special_tokens=True)
