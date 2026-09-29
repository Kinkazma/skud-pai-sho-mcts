import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,MagicMock
from paisho_gen3_controls import Gen3Controls,write,read
class ControlsTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.c=Gen3Controls(self.root/'control.json')
  for name in ('binary','dataset'): (self.root/name).write_text('{}')
  (self.root/'model').write_text('{"generation":"3.2"}')
  self.c.install(self.root/'binary',self.root/'model',self.root/'dataset')
 def tearDown(self):self.tmp.cleanup()
 def test_reference_mix_change_requires_pause_and_keeps_recovery(self):
  with self.assertRaises(ValueError):self.c.set_historical_every(5)
  s=read(self.c.path);s.update(state='paused',pid=None,historical_pool=[{'generation':'Gen3.1'}],remaining_seconds=2270,replay_index='kept');write(self.c.path,s)
  self.c.set_historical_every(5);after=read(self.c.path)
  self.assertEqual(after['historical_every'],5)
  for k in ('model','model_sha256','generation','replay_index','remaining_seconds'):self.assertEqual(after[k],s[k])
  for invalid in (True,0,1,2.5):
   with self.assertRaises(ValueError):self.c.set_historical_every(invalid)
  self.assertEqual(read(self.c.path),after)
 def test_start_freezes_intent_and_rejects_duplicate_click(self):
  with patch('paisho_gen3_controls.subprocess.Popen') as popen:
   popen.return_value.pid=42
   self.c.request('start',2)
   s=read(self.c.path);self.assertEqual(s['remaining_seconds'],7200);self.assertEqual(s['state'],'starting')
   with self.assertRaises(ValueError):self.c.request('start',2)
   self.assertEqual(popen.call_count,1)
 def test_pause_and_stop_have_distinct_remaining_time(self):
  out=self.root/'run';(out/'training').mkdir(parents=True)
  s=read(self.c.path);s.update(state='running',campaign=str(out),end=1000);write(self.c.path,s)
  with patch('paisho_gen3_controls.time.time',return_value=700):self.c.request('pause')
  self.assertEqual(read(self.c.path)['remaining_seconds'],300);self.assertTrue((out/'training/pause-request.json').exists())
  s=read(self.c.path);s['state']='running';write(self.c.path,s);self.c.request('stop')
  self.assertEqual(read(self.c.path)['remaining_seconds'],0);self.assertEqual(read(self.c.path)['state'],'stopping')
 def test_new_session_after_completion_keeps_same_generation_and_model(self):
  s=read(self.c.path);original=s['model'];s.update(state='completed',candidate=original);write(self.c.path,s)
  with patch('paisho_gen3_controls.subprocess.Popen') as popen:
   popen.return_value.pid=42;self.c.request('start',9)
  s=read(self.c.path);self.assertEqual(s['generation'],'3.2');self.assertEqual(s['model'],original)
  self.assertEqual(s['remaining_seconds'],9*3600)
 def test_next_generation_retains_weights_without_starting(self):
  s=read(self.c.path);s.update(state='completed',candidate=s['model']);write(self.c.path,s)
  self.c.request('next');s=read(self.c.path)
  self.assertEqual(s['generation'],'3.3');self.assertEqual(read(s['model'])['generation'],'3.3');self.assertEqual(s['state'],'ready')
 def test_changed_input_or_bad_duration_cannot_start(self):
  for hours in (None,0,float('nan'),25):
   with self.assertRaises((ValueError,TypeError)):self.c.request('start',hours)
  (self.root/'model').write_text('{}')
  with self.assertRaises(ValueError):self.c.request('start',2)
 def test_upgrade_requires_pause_and_preserves_learned_state_and_time(self):
  ref=self.root/'reference';ref.write_text('frozen')
  with self.assertRaises(ValueError):self.c.upgrade_training(self.root/'binary',ref)
  s=read(self.c.path);s.update(state='paused',remaining_seconds=320,replay_index='retained');write(self.c.path,s)
  self.c.upgrade_training(self.root/'binary',ref);after=read(self.c.path)
  for key in ('model','model_sha256','generation','remaining_seconds','replay_index'):self.assertEqual(after[key],s[key])
  self.assertEqual(after['historical_reference'],str(ref.resolve()))
 def test_tactical_upgrade_is_explicit_and_preserves_state(self):
  ref=self.root/'reference';ref.write_text('frozen')
  s=read(self.c.path);s.update(state='paused',remaining_seconds=123,replay_index='old-index');write(self.c.path,s)
  opts={'solver':True,'tactical_positions':0,'correction_replay':True}
  self.c.upgrade_training(self.root/'binary',ref,tactics=opts);after=read(self.c.path)
  self.assertEqual(after['training_options'],opts)
  for key in ('model','generation','remaining_seconds','replay_index'):self.assertEqual(after[key],s[key])
  with self.assertRaises(ValueError):self.c.upgrade_training(self.root/'binary',ref,tactics=opts|{'seconds':0})
  self.assertEqual(read(self.c.path),after)
 def test_changed_frozen_reference_cannot_resume(self):
  from paisho_gen3_controls import digest
  ref=self.root/'reference';ref.write_text('frozen')
  s=read(self.c.path);s.update(state='paused',remaining_seconds=30,historical_reference=str(ref),historical_reference_sha256=digest(ref));write(self.c.path,s)
  ref.write_text('changed')
  with self.assertRaises(ValueError):self.c.request('resume')
  self.assertEqual(read(self.c.path)['state'],'paused')
 def test_gen5_remains_untouched_when_running(self):
  other=MagicMock();other.path=self.root/'other.json';write(other.path,{'campaign':str(self.root/'g5'),'busy':False})
  self.c.gen5=other
  with patch('paisho_gen3_controls.native',return_value=100):
   with self.assertRaises(ValueError):self.c.request('start',2)
  self.assertEqual(read(self.c.path)['state'],'ready');self.assertEqual(read(other.path),{'campaign':str(self.root/'g5'),'busy':False})
if __name__=='__main__':unittest.main()

class RealLifecycleTests(unittest.TestCase):
 def test_detached_pause_resume_stop_keeps_checkpoint_and_drops_time(self):
  import os,time
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);binary=root/'fake-native'
   binary.write_text('''#!/usr/bin/env python3
import json,sys,time
from pathlib import Path
mode=sys.argv[1]
if mode=='defaults':print('{}')
elif mode=='run':
 c=json.loads(Path(sys.argv[2]).read_text());out=Path(c['output']);out.mkdir(parents=True)
 model=out/'model.json';model.write_text(Path(c['model']).read_text());index=out/'replay.json';index.write_text('{}')
 (out/'checkpoint.json').write_text(json.dumps({'model':str(model),'replay_index':str(index)}))
 (out/'progress.json').write_text('{"completed":1,"terminal":1}')
 end=time.time()+c['seconds']
 while time.time()<end and not (out/'pause-request.json').exists() and not (out/'stop-request.json').exists():time.sleep(.02)
elif mode=='human-fit':
 out=Path(sys.argv[4]);out.mkdir();(out/'model.json').write_text(Path(sys.argv[2]).read_text())
''');binary.chmod(0o755)
   model=root/'model';model.write_text('{"generation":"3.2"}');dataset=root/'dataset';dataset.write_text('{}')
   c=Gen3Controls(root/'control.json');c.install(binary,model,dataset)
   def wait(state):
    end=time.monotonic()+15
    while time.monotonic()<end:
     s=read(c.path)
     if s['state']==state:return s
     if s['state']=='failed':self.fail(s.get('error'))
     time.sleep(.05)
    self.fail('timeout '+str(read(c.path)))
   c.request('start',1/60);wait('running')
   # Wait for the native directory before sending a request.
   out=Path(read(c.path)['campaign'])/'training'
   end=time.monotonic()+5
   while not out.exists() and time.monotonic()<end:time.sleep(.02)
   c.request('pause');s=wait('paused');self.assertGreater(s['remaining_seconds'],0);self.assertTrue(Path(s['model']).exists())
   ref=root/'reference';ref.write_text('{}');opts={'solver':True,'tactical_positions':0,'correction_replay':True}
   c.upgrade_training(binary,ref,tactics=opts)
   c.request('resume');wait('running');out=Path(read(c.path)['campaign'])/'training'
   actual=read(out.parent/'config.json')
   for key,value in opts.items():self.assertEqual(actual[key],value)
   end=time.monotonic()+5
   while not out.exists() and time.monotonic()<end:time.sleep(.02)
   c.request('stop');s=wait('stopped');self.assertEqual(s['remaining_seconds'],0);self.assertTrue(Path(s['model']).exists());self.assertIsNone(s.get('error'));self.assertEqual(s['generation'],'3.2');self.assertEqual(len(s['campaign_history']),2)

class CandidateStageTests(ControlsTests):
 def test_stage_34_preserves_archived_time_fifo_and_frozen_pool(self):
  from paisho_gen3_controls import digest
  s=read(self.c.path);s.update(state='paused',remaining_seconds=9144,replay_index='retained',pid=None);write(self.c.path,s)
  model=self.root/'candidate';model.write_text('{"generation":"3.4"}')
  pool=[dict(generation=g,budget=32,path=str(self.root/'model'),sha256=digest(self.root/'model'),solver=False) for g in ('Gen3.1','Gen3.2','Gen3.3')]
  with patch('paisho_gen3_controls.subprocess.check_output',return_value='{"generation":"3.4","value_features":128}'):
   self.c.stage_candidate(self.root/'binary',model,pool)
  new=read(self.c.path);self.assertEqual(new['generation'],'3.4');self.assertEqual(new['replay_index'],'retained');self.assertFalse(new['post_training_fit']);self.assertEqual(new['resource_options']['replay_max_bytes'],48*1024**3)
  self.assertEqual(read(new['previous_control']),s)
  (self.root/'model').write_text('changed')
  with self.assertRaises(ValueError):self.c.request('start',1)
  self.assertEqual(read(self.c.path)['state'],'ready')

class ExtendedCompletionTests(unittest.TestCase):
 def test_extended_run_completes_without_legacy_human_fit(self):
  from paisho_gen3_controls import worker,digest
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);binary=root/'binary';binary.write_text('native');model=root/'model.json';model.write_text('{"generation":"3.4","value128_extra":[0]}');dataset=root/'data';dataset.write_text('{}')
   path=root/'control.json';identity='abcdef0123';write(path,dict(state='starting',action_id=identity,generation='3.4',binary=str(binary),binary_sha256=digest(binary),model=str(model),model_sha256=digest(model),dataset=str(dataset),dataset_sha256=digest(dataset),remaining_seconds=1,post_training_fit=False,historical_every=5,historical_pool=[dict(generation=g,budget=32,path=str(model),sha256=digest(model),solver=False) for g in ('Gen3.1','Gen3.2','Gen3.3')]))
   def launch(args,**kwargs):
    child=MagicMock();child.pid=42;child.wait.return_value=0;child.poll.return_value=0
    if 'run' in args:
     config=read(args[-1]);out=Path(config['output']);out.mkdir();index=out/'replay.json';index.write_text('{}');write(out/'checkpoint.json',dict(model=config['model'],replay_index=str(index)))
    return child
   with patch('paisho_gen3_controls.subprocess.check_output',return_value='{}'),patch('paisho_gen3_controls.subprocess.Popen',side_effect=launch),patch('paisho_gen3_controls.auxiliary') as fit:
    worker(path,identity,'start');fit.assert_not_called()
   state=read(path);self.assertEqual(state['state'],'completed');self.assertEqual(state['remaining_seconds'],0);self.assertEqual(state['candidate'],state['model']);self.assertEqual(state['generation'],'3.4')

   config=read(Path(state['campaign'])/'config.json')
   self.assertEqual(len(config['historical_pool']),3)
   for expert in config['historical_pool']:self.assertEqual(digest(Path(expert['path'])),expert['sha256'])

   self.assertEqual(config['historical_every'],5)
