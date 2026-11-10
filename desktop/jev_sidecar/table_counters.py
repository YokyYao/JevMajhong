"""Read the two numeric counters below the dora panel with the local OCR."""
import re
import unicodedata

import numpy as np
from PIL import ImageOps

# Mahjong Soul: riichi sticks on the left, honba on the right.
DEFAULT_COUNTER_REGION = (.012, .119, .15, .057)


def numbers_after_multiplier(text):
    text = unicodedata.normalize('NFKC', text).replace('×','x').replace('✕','x').replace('X','x')
    text = re.sub(r'(x\s*)[OoＯｏ](?=\s|$)',r'\g<1>0',text)
    return [int(n) for n in re.findall(r'x\s*([0-9]{1,2})(?![0-9])',text)]


def parse_counter_row(output):
    """Use left-to-right order, not OCR's returned text-box order."""
    boxes = sorted(output or [],key=lambda item:min(p[0] for p in item[0]))
    values = numbers_after_multiplier(' '.join(str(item[1]) for item in boxes))
    return {'sticks':values[0],'honba':values[1]} if len(values)==2 else None


def parse_single_counter(output):
    text = ' '.join(str(item[0]) for item in output or [])
    values = numbers_after_multiplier(text)
    if len(values)==1:return values[0]
    text = unicodedata.normalize('NFKC',text).strip()
    return int(text) if re.fullmatch(r'[0-9]{1,2}',text) else None


class TableCounterReader:
    def __init__(self):
        self.values = {'honba':None,'sticks':None}

    def read(self,image,region,get_ocr):
        # A missed or interrupted frame keeps the previous numeric values;
        # it must never manufacture a zero or open a recognition dialog.
        try:
            rect = (region or {}).get('rect',DEFAULT_COUNTER_REGION)
            x,y,w,h = rect
            area = image.convert('RGB').crop((round(x*image.width),round(y*image.height),
                round((x+w)*image.width),round((y+h)*image.height)))
            area = area.rotate(int((region or {}).get('rotation',0)),expand=True)
            if area.width<12 or area.height<6 or np.asarray(area.convert('L')).std()<8:
                return dict(self.values)
            if area.height<56:
                area=area.resize((round(area.width*56/area.height),56))
            ocr=get_ocr()
            output,_=ocr(np.asarray(area),use_cls=False)
            found=parse_counter_row(output)
            if found is None:
                found={}
                # Exclude the red/black stick icons; they can look like digits.
                for i,key in enumerate(('sticks','honba')):
                    left=area.width*i/2; half=area.width/2
                    number=area.crop((round(left+half*.43),0,round(left+half*.99),area.height))
                    for variant in (number,ImageOps.invert(ImageOps.autocontrast(number.convert('L'))).convert('RGB')):
                        text,_=ocr(np.asarray(variant),use_det=False,use_cls=False)
                        value=parse_single_counter(text)
                        if value is not None:
                            found[key]=value;break
            for key,value in found.items():
                if value is not None and 0<=value<=99:self.values[key]=int(value)
        except Exception:
            # Counter OCR is optional to the frame and intentionally silent.
            pass
        return dict(self.values)
