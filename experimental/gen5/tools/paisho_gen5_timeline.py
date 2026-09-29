"""Cached, model-specific evaluation observations; never fit Elo from self-play."""
import json
import math
from pathlib import Path
import time


def read_json(path):
    try:
        with path.open('rb') as stream:
            data = stream.read(2_000_001)
        if len(data) > 2_000_000:
            raise ValueError('oversized evaluation receipt')
        return json.loads(data)
    except FileNotFoundError:
        return {}


def stamp(path):
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        return None


class EloTimeline:
    def __init__(self, campaign):
        self.campaign = Path(campaign)
        self.sources = None
        self.cache = {}
        self.next_read = 0
        self.result = {'measurements': [], 'error': None}

    def read(self, upper):
        if time.monotonic() < self.next_read:
            return self.result
        try:
            if self.sources is None:
                sources, seen, current = [], set(), self.campaign
                while current:
                    current = current.resolve()
                    if current in seen:
                        raise ValueError('campaign ancestry cycle')
                    seen.add(current); sources.append(current)
                    config = read_json(current/'config.json')
                    prior = read_json(Path(config['resume_progress'])) if config.get('resume_progress') else {}
                    current = Path(prior['previous_campaign']) if prior.get('previous_campaign') else None
                self.sources = sources
            observations = {}
            for campaign in self.sources:
                for index in range(upper):
                    directory = campaign/'training/history'/f'sweep-{index:04}'
                    assessment_path = directory/'assessment.json'
                    for reference in ('initial', 'gen3-1'):
                        report_path = directory/reference/'report.json'
                        report_stamp = stamp(report_path)
                        if report_stamp is None:
                            continue
                        plan_path = directory/reference/'plan.json'
                        signature = (report_stamp, stamp(assessment_path), stamp(plan_path))
                        key = (campaign, index, reference)
                        cached = self.cache.get(key)
                        if cached is None or cached[0] != signature:
                            report = read_json(report_path)
                            assessment = read_json(assessment_path)
                            plan = read_json(plan_path)
                            wins, draws, losses = (report.get(k, 0) for k in ('wins', 'draws', 'losses'))
                            observation = {
                                'index': index, 'reference': reference,
                                'model': plan.get('candidate', assessment.get('model')),
                                'anchor': plan.get('reference_sha256'),
                                'version': assessment.get('version'),
                                'published': report_stamp[0]/1e9,
                                'elo': 400*math.log10((wins+.5*draws+1)/(losses+.5*draws+1)) if wins+draws+losses else None,
                                'wins': wins, 'draws': draws, 'losses': losses,
                                'unknown': report.get('unknown', 0),
                                'complete_pairs': report.get('complete_pairs', 0),
                                'simulations': plan.get('options', {}).get('simulations'),
                                'decision_limit': plan.get('options', {}).get('decisions'),
                            }
                            self.cache[key] = (signature, observation)
                        row = self.cache[key][1]
                        identity = (index, reference, row['model'], row['anchor'])
                        old = observations.get(identity)
                        # Resumed runs copy reports. Keep their original publication,
                        # not the time a copy happened to be made.
                        if old is None or row['published'] < old['published']:
                            observations[identity] = row
            self.result = {'measurements': sorted(observations.values(), key=lambda r: r['published']), 'error': None}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.result = {**self.result, 'error': str(exc)}
        self.next_read = time.monotonic()+30
        return self.result
