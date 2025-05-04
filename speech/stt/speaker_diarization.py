from pyannote.audio import Pipeline
from typing import List, Dict

class SpeakerDiarization:
    def __init__(self):
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        
    def process_audio(self, audio_file: str) -> List[Dict]:
        """
        Process audio file and identify different speakers
        Returns list of segments with speaker IDs and timestamps
        """
        diarization = self.pipeline(audio_file)
        
        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = {
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            }
            results.append(segment)
            
        return results
