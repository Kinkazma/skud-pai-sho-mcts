"""Two explicit training recipes sharing the latest durable Gen5 weights."""
MODES=('selfplay','gen3-replay')
def apply(options,mode):
    if mode not in MODES:raise ValueError('Unknown training mode.')
    if not options.get('case_curriculum'):raise ValueError('Human case curriculum required.')
    # Campaign choice, distinct from backward-compatible native config defaults.
    options.setdefault('minimum_search_depth',5)
    options.update(legacy_replay=mode=='gen3-replay',control_stop=True,
                   budgets=[[256,.5],[512,.5]],checkpoint_fraction=0,
                   checkpoint_seconds=30,inline_learning=True)
    if options.get('opponents'):
        # Both duration controls continue the user-selected 80/20 curriculum.
        options.update(legacy_replay=True,historical=False,history_interval=0,
                       reuse_idle_secondary=False,historical_replay_fraction=0)
        return options
    if mode=='gen3-replay':
        options.update(historical=False,history_interval=0,reuse_idle_secondary=False,
                       historical_replay_fraction=0)
    else:
        options.update(historical=True,history_interval=900,reuse_idle_secondary=True,
                       secondary_threads=min(2,max(1,options.get('threads',10)-1)),
                       historical_capacity_fraction=.075,historical_replay_fraction=.05)
    return options
