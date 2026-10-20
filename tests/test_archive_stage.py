import hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kaggle'))
from archive_stage import stage

class ArchiveStageTests(unittest.TestCase):
    def run_case(self,kind,max_seconds=20):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);b=io.BytesIO();Image.new('RGB',(8,12)).save(b,format='PNG');data=b.getvalue()
            name='../escape.png' if kind=='unsafe' else 'PETA/images/example.png'
            z=p/'sample.upar'
            with zipfile.ZipFile(z,'w') as f:
                f.writestr(name,data);f.writestr('ARCHIVE_MANIFEST.json',json.dumps({name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}))
            h=hashlib.sha256(z.read_bytes()).hexdigest()
            item={'path':str(z),'archive_bytes':z.stat().st_size,'archive_sha256':('0'*64 if kind=='bad_hash' else h)}
            if kind=='good':
                r=stage([item],p/'data',p/'report.json',max_seconds,10,20,1)
                self.assertTrue(r['passed']);self.assertEqual(r['images'],1)
            else:
                with self.assertRaises(RuntimeError):stage([item],p/'data',p/'report.json',max_seconds,10,20,1)
                self.assertFalse(json.loads((p/'report.json').read_text())['passed'])
            self.assertFalse((p/'escape.png').exists())
    def test_valid(self):self.run_case('good')
    def test_archive_hash(self):self.run_case('bad_hash')
    def test_path_escape(self):self.run_case('unsafe')
    def test_watchdog(self):self.run_case('timeout',.001)

if __name__=='__main__':unittest.main()
