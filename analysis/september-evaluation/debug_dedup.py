import sys
import runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'backend'))
from app.services.pipeline import RiderTrackManager, association_signature_score
old_update = RiderTrackManager.update
old_final = RiderTrackManager.finalize_pending_track
def update(self, analysis, number, frame=None):
    before=dict(self.track_aliases)
    old_update(self,analysis,number,frame)
    if before!=self.track_aliases:
        print('ALIASES',number,self.track_aliases,flush=True)
def finalize(self, track):
    print('FINALIZE',track['id'],track['pending_frame_number'],
          [(s['track_ids'],s['frame_number'],association_signature_score(s,track['pending_association'])) for s in self.saved_violation_signatures],flush=True)
    result=old_final(self,track)
    print('SAVED',bool(result),flush=True)
    return result
RiderTrackManager.update=update
RiderTrackManager.finalize_pending_track=finalize
sys.argv=['run_benchmark.py','--v3','--output','v3-debug','--intervals','0.5','--clips','IMG_7082.MOV']
runpy.run_path(str(Path(__file__).with_name('run_benchmark.py')),run_name='__main__')
