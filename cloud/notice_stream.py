"""Seekable safetensors view with a modification notice; tensor payload bytes never change."""
import io
import json
import os
from pathlib import Path
import struct


class NoticeStream(io.RawIOBase):
    def __init__(self,path,notice):
        super().__init__()
        self.file=Path(path).open('rb')
        raw=self.file.read(8)
        if len(raw)!=8:
            self.file.close();raise ValueError('Truncated safetensors file')
        n=struct.unpack('<Q',raw)[0]
        if not 2<=n<=min(100_000_000,os.fstat(self.file.fileno()).st_size-8):
            self.file.close();raise ValueError('Invalid safetensors header')
        try:
            header=json.loads(self.file.read(n))
            metadata=header.setdefault('__metadata__',{})
            if not isinstance(notice,str) or not isinstance(metadata,dict) or any(not isinstance(v,str) for v in metadata.values()):
                raise ValueError('Invalid safetensors metadata')
            metadata['modification_notice']=notice
            encoded=json.dumps(header,separators=(',',':'),ensure_ascii=False).encode()
            encoded+=b' '*((-len(encoded))%8)
        except BaseException:
            self.file.close()
            raise
        self.prefix=struct.pack('<Q',len(encoded))+encoded
        self.payload_offset=8+n
        self.size=len(self.prefix)+os.fstat(self.file.fileno()).st_size-self.payload_offset
        self.position=0

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position

    def seek(self,offset,whence=os.SEEK_SET):
        self._checkClosed()
        target=offset if whence==os.SEEK_SET else self.position+offset if whence==os.SEEK_CUR else self.size+offset if whence==os.SEEK_END else None
        if target is None or target<0:raise ValueError('Invalid seek')
        self.position=target
        return target

    def readinto(self,buffer):
        self._checkClosed()
        view=memoryview(buffer).cast('B')
        wanted=min(len(view),max(0,self.size-self.position))
        done=0
        if self.position<len(self.prefix):
            count=min(wanted,len(self.prefix)-self.position)
            view[:count]=self.prefix[self.position:self.position+count]
            self.position+=count;done+=count
        if done<wanted:
            self.file.seek(self.payload_offset+self.position-len(self.prefix))
            count=self.file.readinto(view[done:wanted])
            if count==0:raise OSError('Source payload was truncated during reading')
            self.position+=count;done+=count
        return done

    def close(self):
        if not self.closed:self.file.close()
        super().close()


def open_notice(path,notice):
    return io.BufferedReader(NoticeStream(path,notice),buffer_size=8*1024*1024)
