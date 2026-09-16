"""Apply explicit visual-review decisions and export normalized YOLO labels."""
import hashlib
import html
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'analysis/train-video-footage-staging'
OUT=ROOT/'analysis/train-video-footage'
SOURCE=Path('C:/Users/ADMIN/Downloads/Sp vd')
NAMES=['Helmet','License Plate','Motorcycle','No Helmet','No License Plate','Person']
COLORS=['#00bbff','#ffee00','#ff8800','#ff2244','#dd55ff','#55ee66']

# Proposal indices retained after source-image review. Distant/uncertain
# background objects are excluded spatially by the final contextual crop.
KEEP={
0:[2,3,4,5],1:[1,2,6,7,9,13],2:[0,5,7,9],3:[0,1,3,4],4:[0,2,3,4],
7:list(range(11)),8:[1,2,3,4,5],9:[1,2,3,6,7,9],11:[0,1,2],12:[0,3,4,5],
13:[0,2,4,5],14:[0,1,2,3],15:[0,1,2,3],16:[0,1,2,3,4],17:[0,1,2,3],
18:[0,2,3,4],20:[0,1,2,3],22:[0,1,2,3],24:[0,1,2,3,4,5],25:[0,1,2,3],
26:[0,2,3,4,5],27:list(range(6)),28:[1,3,4],29:[0,1,3,4],30:[2,3,4],31:[0,1],
32:[0,2,3,4],33:[0,1,2,3],34:[0,1,2,3],35:[0,1,2],36:[0,1,2,3,4],37:[0,1,2,3,4],
39:[1,2,3,4],40:[0,1,2,3],41:[1,2],42:[0,1,4,5,6,7],44:[0,3,4,5,6],
45:[0,2,3,4],46:[1,4,7,8],48:[1,4,5],49:[1,2,3],51:[0,3,5,7],53:[0,2,5,6],
54:list(range(6)),55:list(range(5)),56:[0,4,7,9],57:[0,2,4,5,7],58:list(range(8)),
59:[0,1,3,4,5,6,7],60:[0,1,2,3],61:[0,3,4,7],62:[0,1,2,3,4,5,6,8],63:[0,2,4,8,9,11],
}
# Replacements: proposal index -> corrected (class, xyxy) in road-crop pixels.
REPLACE={
0:{2:(0,[629,425,754,553]),4:(2,[556,553,900,1056]),5:(5,[565,425,865,950])},
8:{3:(5,[474,331,884,835])},
12:{5:(5,[527,307,770,697])},
16:{0:(3,[925,189,1012,265])},
25:{1:(2,[260,313,550,787]),3:(5,[315,174,572,638])},
26:{3:(5,[550,202,900,701]),5:(5,[509,192,777,687])},
27:{3:(5,[644,165,820,470])},
30:{2:(2,[367,323,729,817]),3:(5,[364,225,640,660]),4:(5,[414,230,742,733])},
31:{0:(2,[206,423,659,1040]),1:(5,[256,247,651,906])},
36:{1:(3,[853,158,901,208])},
37:{3:(5,[598,167,785,535])},
39:{3:(5,[614,198,846,535])},
41:{2:(5,[604,247,944,732])},
44:{3:(0,[549,204,593,270]),5:(5,[486,169,733,543]),6:(2,[448,292,725,650])},
56:{0:(3,[848,176,901,248]),9:(5,[757,176,944,501])},
57:{5:(5,[423,173,493,363])},
58:{5:(5,[424,202,615,537]),6:(2,[667,318,1047,854])},
59:{4:(5,[208,221,444,579]),5:(2,[240,367,467,668])},
}
# Missing visible heads, plates and individually occluded people, added by eye.
ADD={
8:[(0,[624,373,708,451])],
12:[(0,[634,307,685,384]),(5,[622,307,750,629])],
16:[(3,[882,177,937,261])],
25:[(0,[390,209,461,288]),(3,[455,174,530,253])],
26:[(1,[729,698,815,780])],
28:[(1,[814,522,877,571])],
30:[(0,[518,225,614,324]),(3,[623,230,700,334]),(1,[605,619,697,692]),(1,[697,319,748,347])],
31:[(0,[352,293,405,391]),(3,[391,247,515,385]),(1,[467,790,596,880]),(5,[294,295,406,548])],
32:[(0,[594,198,623,251]),(5,[524,198,624,473])],
35:[(0,[799,181,869,254])],
36:[(5,[778,163,934,418])],
37:[(5,[573,193,695,529])],
39:[(5,[554,275,666,526])],
41:[(3,[668,233,744,309]),(3,[708,249,793,327]),(5,[553,233,744,733]),(1,[786,681,892,762])],
44:[(5,[466,204,590,563])],
46:[(3,[759,183,798,226])],
48:[(1,[666,587,753,648])],
49:[(3,[478,184,538,249]),(0,[462,217,485,265]),(5,[382,221,495,582])],
55:[(5,[681,170,773,374])],
56:[(0,[825,182,865,246]),(5,[750,182,865,484])],
57:[(0,[493,215,523,254]),(5,[430,216,525,490]),(5,[407,216,461,359]),(2,[403,269,497,396])],
58:[(1,[930,539,1017,589]),(5,[415,205,536,504])],
59:[(1,[352,439,425,466])],
}

# Second visual pass: include the visible handlebars/mirrors as well as the
# rear wheel/body in motorcycle boxes, instead of teaching rear-only boxes.
BIKE_EXTENTS={
4:{4:[440,424,773,801]},8:{4:[485,475,804,990]},9:{7:[652,426,883,701]},
11:{1:[711,385,899,652]},12:{4:[514,434,778,807]},14:{3:[820,480,960,653]},
16:{3:[815,270,1044,559]},18:{4:[506,294,774,699]},20:{2:[223,319,506,799]},
22:{3:[326,328,593,695]},26:{4:[504,315,845,837]},27:{4:[620,258,857,536]},
28:{3:[578,280,881,671]},29:{4:[625,290,845,595]},33:{2:[573,321,887,771]},
34:{3:[450,370,703,662]},36:{4:[738,238,954,497]},39:{4:[548,344,811,667]},
40:{3:[267,254,389,453]},41:{1:[548,423,919,903]},45:{4:[735,273,927,501]},
48:{5:[452,348,768,866]},49:{2:[358,310,588,678]},51:{7:[616,329,902,764]},
54:{4:[752,278,1024,637]},56:{7:[755,277,965,588]},57:{7:[426,289,589,560]},
60:{3:[790,314,976,572]},62:{5:[275,325,586,826]},63:{8:[510,298,708,628]},
}
for idx,edits in BIKE_EXTENTS.items():
    REPLACE.setdefault(idx,{}).update({bi:(2,box) for bi,box in edits.items()})
ADD[57][-1]=(2,[403,236,497,396])
# Incidental visible car plates within retained contextual crops.
ADD.setdefault(4,[]).append((1,[683,416,715,453]))
ADD.setdefault(15,[]).append((1,[278,277,333,304]))
ADD.setdefault(49,[]).append((1,[370,232,410,253]))


def main():
    proposals=json.loads((STAGE/'proposals.json').read_text())
    OUT.mkdir(exist_ok=True)
    for folder in ['images','labels','annotated','review','source_frames']:(OUT/folder).mkdir(exist_ok=True)
    manifest=[];counts=Counter();tiles=[];decisions=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for i,rec in enumerate(proposals):
        if i not in KEEP:
            decisions.append(dict(candidate_index=i,image=rec['image'],decision='excluded',reason='Incomplete edge view, unhelpful duplicate/distant view, or no clear motorcycle in candidate'))
            continue
        boxes=[]
        for bi in KEEP[i]:
            b=deepcopy(rec['boxes'][bi]);b['origin']='model_proposal_visually_reviewed';b['proposal_index']=bi
            if bi in REPLACE.get(i,{}):
                b['class_id'],b['xyxy']=REPLACE[i][bi];b['origin']='visually_corrected'
            boxes.append(b)
        for cls,xyxy in ADD.get(i,[]):boxes.append(dict(class_id=cls,xyxy=xyxy,origin='visually_added'))
        road=Image.open(STAGE/'images'/rec['image']).convert('RGB')
        # Keep context without retaining distant unscorable riders as negatives.
        roi=[max(0,min(b['xyxy'][0] for b in boxes)-40),max(0,min(b['xyxy'][1] for b in boxes)-40),
             min(road.width,max(b['xyxy'][2] for b in boxes)+40),min(road.height,max(b['xyxy'][3] for b in boxes)+40)]
        im=road.crop(roi)
        for b in boxes:
            x1,y1,x2,y2=b['xyxy'];b['xyxy']=[x1-roi[0],y1-roi[1],x2-roi[0],y2-roi[1]]
        # Re-decode original pixels for final crop and preserve the full frame.
        cap=cv2.VideoCapture(str(SOURCE/rec['source_clip']));cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1)
        cap.set(cv2.CAP_PROP_POS_FRAMES,rec['frame_number']);ok,frame=cap.read();cap.release();assert ok
        full=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        full.save(OUT/'source_frames'/rec['image'],quality=97)
        source_crop=[roi[0],roi[1]+rec['crop_xyxy'][1],roi[2],roi[3]+rec['crop_xyxy'][1]]
        im=full.crop(source_crop);im.save(OUT/'images'/rec['image'],quality=97)
        rows=[]
        for b in boxes:
            x1,y1,x2,y2=b['xyxy'];assert 0<=x1<x2<=im.width and 0<=y1<y2<=im.height,(i,b,im.size)
            cls=b['class_id'];counts[NAMES[cls]]+=1
            rows.append(f'{cls} {(x1+x2)/2/im.width:.7f} {(y1+y2)/2/im.height:.7f} {(x2-x1)/im.width:.7f} {(y2-y1)/im.height:.7f}')
        (OUT/'labels'/Path(rec['image']).with_suffix('.txt')).write_text('\n'.join(rows)+'\n')
        draw=ImageDraw.Draw(im)
        for b in boxes:
            x1,y1,x2,y2=b['xyxy'];color=COLORS[b['class_id']];name=NAMES[b['class_id']]
            draw.rectangle((x1,y1,x2,y2),outline=color,width=2)
            ty=max(0,y1-19);tw=int(draw.textlength(name,font=font))+4
            draw.rectangle((x1,ty,min(im.width,x1+tw),ty+19),fill='black')
            draw.text((x1+2,ty),name,font=font,fill=color)
        im.save(OUT/'annotated'/rec['image'],quality=96)
        im.thumbnail((395,510));tile=Image.new('RGB',(400,540),'white');tile.paste(im,((400-im.width)//2,30))
        ImageDraw.Draw(tile).text((4,4),f"#{i} {rec['image']}",font=font,fill='black');tiles.append(tile)
        item={k:v for k,v in rec.items() if k not in ['boxes','review_status','crop_xyxy','image_size','image_sha256']}
        item.update(candidate_index=i,crop_xyxy=source_crop,image_size=[roi[2]-roi[0],roi[3]-roi[1]],boxes=boxes,
                    image_sha256=hashlib.sha256((OUT/'images'/rec['image']).read_bytes()).hexdigest(),
                    review_status='AI_visually_reviewed_and_corrected',split_group=rec['source_clip'])
        manifest.append(item)
        decisions.append(dict(candidate_index=i,image=rec['image'],decision='included',retained_proposals=KEEP[i],replacements=REPLACE.get(i,{}),added_boxes=ADD.get(i,[])))
    for start in range(0,len(tiles),8):
        sheet=Image.new('RGB',(1600,1080),'#ddd')
        for j,tile in enumerate(tiles[start:start+8]):sheet.paste(tile,(j%4*400,j//4*540))
        sheet.save(OUT/'review'/f'review-{start//8:02d}.jpg',quality=96)
    summary=dict(source_clips=len({r['source_clip'] for r in manifest}),candidate_frames=len(proposals),
                 training_images=len(manifest),excluded_frames=len(proposals)-len(manifest),
                 boxes=sum(counts.values()),class_counts={n:counts[n] for n in NAMES},
                 annotation_method='V4/YOLO11s proposals followed by AI visual review, corrections and additions; not independently human-validated',
                 image_scope='Contextual motorcycle crops; full source frames preserved separately and not labeled for training',
                 evaluation_overlap='All source clips were previously used in the Sp vd benchmark. Do not use that benchmark as an independent test after training on this set.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT/'review/decisions.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/'classes.txt').write_text('\n'.join(NAMES)+'\n')
    (OUT/'notes.json').write_text(json.dumps({'categories':[dict(id=i,name=n) for i,n in enumerate(NAMES)]},indent=2))
    cards='\n'.join(f'<article><h3>{html.escape(r["image"])}</h3><p>{r["source_clip"]} · {r["timestamp"]:.2f}s · {len(r["boxes"])} boxes</p><a href="annotated/{r["image"]}"><img src="annotated/{r["image"]}" loading="lazy"></a><p><a href="images/{r["image"]}">Clean training crop</a> · <a href="source_frames/{r["image"]}">Full source frame</a></p></article>' for r in manifest)
    (OUT/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>SafeRide training annotations</title><style>body{font:16px system-ui;background:#eee;padding:20px}article{background:white;padding:12px;margin:8px;display:inline-block;vertical-align:top;width:340px}img{max-width:100%;max-height:530px}</style><h1>train-video-footage</h1><p>AI visually reviewed annotations. Use images/ and labels/ for training; previews and full source frames are for review.</p>'+cards,encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
