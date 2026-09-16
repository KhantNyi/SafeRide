"""Render indexed proposals for explicit visual annotation review."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/train-video-footage-staging'
COLORS={0:'#00bbff',1:'#ffee00',2:'#ff8800',3:'#ff2244',4:'#dd55ff',5:'#55ee66'}
NAMES=['H','LP','M','NH','NLP','P']
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
data=json.loads((OUT/'proposals.json').read_text())
tiles=[]
for idx,rec in enumerate(data):
    im=Image.open(OUT/'images'/rec['image']).convert('RGB')
    draw=ImageDraw.Draw(im)
    for bi,b in enumerate(rec['boxes']):
        box=b['xyxy'];color=COLORS[b['class_id']]
        draw.rectangle(box,outline=color,width=2)
        text=f"{bi}:{NAMES[b['class_id']]}"
        x,y=box[:2];y=max(0,y-20)
        draw.rectangle((x,y,x+len(text)*12,y+20),fill='black')
        draw.text((x,y),text,font=font,fill=color)
    im.save(OUT/'annotated'/rec['image'],quality=95)
    im.thumbnail((535,525))
    tile=Image.new('RGB',(540,555),'white');tile.paste(im,(0,30))
    ImageDraw.Draw(tile).text((4,4),f"#{idx} {rec['image']}",fill='black',font=font)
    tiles.append(tile)
for start in range(0,len(tiles),6):
    sheet=Image.new('RGB',(1620,1110),'#ddd')
    for i,tile in enumerate(tiles[start:start+6]):sheet.paste(tile,(i%3*540,i//3*555))
    sheet.save(OUT/'review'/f'sheet-{start//6:02d}.jpg',quality=96)
print(len(data),'frames;',sum(len(r['boxes']) for r in data),'proposed boxes')
