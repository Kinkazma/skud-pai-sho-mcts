import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from prepare_gen5_resume import prepare


class ResumeTests(unittest.TestCase):
    def native_fixture(self,root):
        campaign=root/'campaign'
        for name in ['bin','training/games','training/models','training/history']:(campaign/name).mkdir(parents=True)
        def write(path,value):path.write_text(json.dumps(value))
        binary=campaign/'bin/paisho-gen5';binary.write_bytes(b'frozen')
        write(campaign/'plan.json',{'hashes':{'bin/paisho-gen5':hashlib.sha256(b'frozen').hexdigest()}})
        prior={'completed':10,'next_game_id':11,'next_history_index':4,'seat_results':{'Historical':dict(games=10,wins=6,losses=4,draws=0,unresolved_seat=0)}}
        write(campaign/'prior.json',prior)
        write(campaign/'config.json',{'seed':1,'resume_progress':str(campaign/'prior.json'),'checkpoint_seconds':30,'replay_max_bytes':100})
        # No payload file is created: preparation must not reload the FIFO.
        source={'path':str(root/'native-payload.targets'),'sha256':'a'*64}
        replay={'schema':'paisho-gen5-replay-index-v1','rules':'skud-pai-sho-gen5-v1','rows':[{'source':source,'indices':[5,2],'lane':'Historical'}]}
        index=campaign/'training/replay-checkpoint.index.json';index.write_text(json.dumps(replay,indent=3)+'\n')
        model=campaign/'training/models/model-0000012.json';model.write_text('{\n "updates":40, "parameters":[-0.0,1.0]\n}\n')
        progress={'completed':11,'version':12,'durable_version':12,'updates':40,'last_game_id':11,'replay_positions':2,'replay_bytes':1000,'next_history_index':7,
                  'next_game_id':22,'checkpoint_model':str(model),'checkpoint_replay_index':str(index),'learner_rng':98765,'recall_quotas':{'credit':.5},
                  'publication_guard':{'learning_loop_v3':True,'learned_choices':{'payload':{'active':[1,2],'pending':[3]}}},'case_states':{'0':{'id':9}},'legacy_ladder':{'stage':2}}
        write(campaign/'training/durable-progress.json',progress)
        write(campaign/'training/progress.json',{**progress,'updates':999,'version':999})
        row={'id':11,'model_version':12,'lane':'Historical','candidate_seat':'Guest','outcome':'Win(Guest)','eligible_examples':9,'targets_file':'never-read','targets_sha256':'b'*64}
        (campaign/'training.log').write_text(json.dumps(row)+'\n')
        return campaign,progress,model,index

    def test_native_fifo_and_model_bytes_win_over_receipt_reconstruction(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);campaign,progress,model,index=self.native_fixture(root)
            model_before=model.read_bytes();index_before=index.read_bytes()
            out=prepare(campaign,root/'prepared')
            self.assertEqual((out/'model.json').read_bytes(),model_before)
            self.assertEqual((out/'replay.index.json').read_bytes(),index_before)
            resumed=json.loads((out/'resume-progress.json').read_text())
            for key,value in progress.items():self.assertEqual(resumed[key],value,key)
            self.assertEqual(resumed['previous_campaign'],str(campaign.resolve()))
            self.assertEqual(resumed['seat_results']['Historical']['wins'],7)
            receipt=json.loads((out/'receipt.json').read_text())
            self.assertTrue(receipt['native_state_exact']);self.assertFalse(receipt['payloads_reloaded_by_preparation'])
            self.assertEqual(receipt['model_sha256'],hashlib.sha256(model_before).hexdigest())
            self.assertEqual(receipt['replay_index_sha256'],hashlib.sha256(index_before).hexdigest())
            self.assertEqual(model.read_bytes(),model_before);self.assertEqual(index.read_bytes(),index_before)

    def test_named_native_checkpoint_failure_never_falls_back(self):
        for fault in ['missing','malformed','count','updates','version']:
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as d:
                root=Path(d);campaign,progress,model,index=self.native_fixture(root)
                if fault=='missing':index.unlink()
                elif fault=='malformed':index.write_text('{}')
                elif fault=='count':progress['replay_positions']=3
                elif fault=='updates':model.write_text('{"updates":41}')
                elif fault=='version':progress['durable_version']=11
                (campaign/'training/durable-progress.json').write_text(json.dumps(progress))
                with self.assertRaises((ValueError,FileNotFoundError)):prepare(campaign,root/'rejected')
                self.assertFalse((root/'rejected').exists())

    def test_changed_native_checkpoint_hash_is_rejected_before_output(self):
        import prepare_gen5_resume as recovery
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);campaign,_,_,index=self.native_fixture(root);real_digest=recovery.digest
            with patch.object(recovery,'digest',side_effect=lambda p:'0'*64 if p==index else real_digest(p)):
                with self.assertRaisesRegex(ValueError,'checkpoint changed'):prepare(campaign,root/'rejected')
            self.assertFalse((root/'rejected').exists())

    def test_native_checkpoint_without_new_games_preserves_saved_state(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);campaign,progress,_,index=self.native_fixture(root)
            progress['completed']=10;progress['seat_results']={'Historical':{'games':99,'wins':80}}
            (campaign/'training/durable-progress.json').write_text(json.dumps(progress))
            (campaign/'training.log').unlink()
            out=prepare(campaign,root/'prepared')
            resumed=json.loads((out/'resume-progress.json').read_text())
            self.assertEqual(resumed['seat_results'],progress['seat_results'])
            self.assertEqual(resumed['next_history_index'],7)
            self.assertEqual((out/'replay.index.json').read_bytes(),index.read_bytes())

    def test_actions_uses_exact_final_capture_model_and_index_in_new_config(self):
        from paisho_gen5_actions import Actions,read
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);campaign,progress,model,index=self.native_fixture(root)
            options=read(campaign/'config.json')
            options.update(control_stop=True,learning_loop_v3=True,publication_transfer=True,
                           legacy_replay=True,recall_fraction=.5,case_curriculum={'archive':'unchanged'},
                           opponents=[{'generation':'3.1'}])
            (campaign/'config.json').write_text(json.dumps(options))
            remaining=6753.509285926819
            actions=Actions(campaign);state=read(actions.path)
            state.update(paused=True,busy=True,job_pid=os.getpid(),remaining_seconds=remaining)
            actions.save(state)
            final_model=campaign/'training/models/model-0000013.json'
            final_index=campaign/'training/replay-final-checkpoint.index.json'
            final={**progress,'version':13,'durable_version':13,'updates':60,
                   'checkpoint_model':str(final_model),'checkpoint_replay_index':str(final_index)}
            expected_model=b'{"updates":60,"parameters":[-0.0,2.0]}\n'
            expected_index=index.read_bytes().replace(b'5,',b'8,')
            def capture(*_,**kwargs):
                final_model.write_bytes(expected_model);final_index.write_bytes(expected_index)
                for name in ['progress.json','durable-progress.json']:(campaign/'training'/name).write_text(json.dumps(final))
                row=json.loads((campaign/'training.log').read_text());row['model_version']=13
                (campaign/'training.log').write_text(json.dumps(row)+'\n')
                return {'pid':42,'kind':'cooperative-native-drain','version':13,'updates':60,'completed':11}
            def launch(config,binary,out,end,**kwargs):
                deployed=read(config);resumed=read(deployed['resume_progress'])
                self.assertEqual(Path(deployed['model']).name,'model.json')
                self.assertEqual(Path(deployed['model']).parent,Path(deployed['replay_index']).parent)
                self.assertEqual(Path(deployed['model']).read_bytes(),expected_model)
                self.assertEqual(Path(deployed['replay_index']).read_bytes(),expected_index)
                self.assertEqual(resumed['updates'],60)
                self.assertEqual(resumed['publication_guard'],progress['publication_guard'])
                self.assertEqual(resumed['recall_quotas'],progress['recall_quotas'])
                self.assertTrue(deployed['learning_loop_v3']);self.assertTrue(deployed['publication_transfer'])
                out.mkdir();return {'output':str(out)}
            with patch('paisho_gen5_actions.native',return_value=42),patch('paisho_gen5_actions.subprocess.check_output',return_value='T'),patch('paisho_gen5_actions.capture_native',side_effect=capture),patch('paisho_gen5_actions.start',side_effect=launch),patch('paisho_gen5_actions.os.kill') as kill:
                actions.continue_run(remaining/3600)
                kill.assert_not_called()
            self.assertAlmostEqual(read(actions.path)['remaining_seconds'],remaining)

    def test_second_continuation_keeps_initial_replay_and_cumulative_seats(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);campaign=root/'campaign'
            for name in ['bin','training/games','training/models','training/history']:(campaign/name).mkdir(parents=True)
            def write(path,value):path.write_text(json.dumps(value))
            binary=campaign/'bin/paisho-gen5';binary.write_bytes(b'frozen')
            write(campaign/'plan.json',{'hashes':{'bin/paisho-gen5':hashlib.sha256(b'frozen').hexdigest()}})
            prior={'completed':10,'next_game_id':11,'next_history_index':4,'seat_results':{'Historical':dict(games=10,wins=6,losses=4,draws=0,unresolved_seat=0)}}
            write(campaign/'prior.json',prior)
            old=root/'old.targets';old.write_bytes(b'old')
            write(campaign/'initial.index.json',{'rows':[{'source':{'path':str(old),'sha256':hashlib.sha256(b'old').hexdigest()},'indices':[2,3,4,5],'lane':'Selfplay'}]})
            write(campaign/'config.json',{'seed':1,'resume_progress':str(campaign/'prior.json'),'replay_index':str(campaign/'initial.index.json')})
            target=campaign/'training/games/new.targets';target.write_bytes(b'new')
            row={'id':11,'model_version':12,'lane':'Historical','candidate_seat':'Guest','outcome':'Win(Guest)','eligible_examples':2,'targets_file':target.name,'targets_sha256':hashlib.sha256(b'new').hexdigest()}
            (campaign/'training.log').write_text(json.dumps(row)+'\n')
            write(campaign/'training/progress.json',{'completed':11,'version':12,'updates':40,'last_game_id':11,'replay_positions':4,'replay_bytes':100})
            write(campaign/'training/models/model-0000012.json',{'updates':40})
            output=prepare(campaign,root/'prepared')
            index=json.loads((output/'replay.index.json').read_text())
            self.assertEqual([r['indices'] for r in index['rows']],[[4,5],[0,1]])
            state=json.loads((output/'resume-progress.json').read_text())
            self.assertEqual(state['seat_results']['Historical']['wins'],7)
            self.assertEqual(state['completed'],11)
            self.assertEqual(state['next_history_index'],4)

    def test_fifo_recovery_uses_receipt_order_hashes_and_partial_first_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);campaign=root/'campaign'
            for name in ['bin','training/games','training/models','training/history']:(campaign/name).mkdir(parents=True)
            binary=campaign/'bin/paisho-gen5';binary.write_bytes(b'frozen')
            def write(path,value):path.write_text(json.dumps(value))
            write(campaign/'config.json',{'seed':1})
            write(campaign/'plan.json',{'hashes':{'bin/paisho-gen5':hashlib.sha256(binary.read_bytes()).hexdigest()}})
            rows=[]
            for i in [8,3]:
                target=campaign/'training/games'/f'game-{i}.targets.json.gz';target.write_bytes(b'archived targets')
                rows.append({'id':i,'model_version':2,'lane':'Historical','candidate_seat':'Guest','outcome':'Win(Guest)',
                             'eligible_examples':3,'targets_file':target.name,'targets_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
            (campaign/'training.log').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            write(campaign/'training/progress.json',{'completed':2,'version':2,'updates':4,'last_game_id':3,'replay_positions':4,'replay_bytes':100})
            write(campaign/'training/models/model-0000002.json',{'updates':4})
            output=prepare(campaign,root/'prepared')
            index=json.loads((output/'replay.index.json').read_text())
            self.assertEqual([r['indices'] for r in index['rows']],[[2],[0,1,2]])
            state=json.loads((output/'resume-progress.json').read_text())
            self.assertEqual(state['next_game_id'],9)
            self.assertEqual(state['seat_results']['Historical']['wins'],2)
            # A live RAM publication beyond the durable checkpoint is ignored.
            config=json.loads((campaign/'config.json').read_text());config['checkpoint_seconds']=30
            write(campaign/'config.json',config)
            durable=json.loads((campaign/'training/progress.json').read_text())
            write(campaign/'training/durable-progress.json',durable)
            write(campaign/'training/progress.json',{**durable,'version':999,'completed':3})
            with (campaign/'training.log').open('a') as stream:stream.write(json.dumps({**rows[-1],'id':99,'model_version':999})+'\n')
            recovered=prepare(campaign,root/'from-checkpoint')
            self.assertEqual(json.loads((recovered/'resume-progress.json').read_text())['version'],2)
            target.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):prepare(campaign,root/'rejected')
            self.assertFalse((root/'rejected').exists())


if __name__=='__main__':unittest.main()
