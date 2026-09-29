import unittest
from paisho_throughput import CounterRates


class CounterTests(unittest.TestCase):
    def test_resume_subtracts_inherited_games_and_elapsed(self):
        r=CounterRates();a=r.observe(1020,{'Selfplay':506,'Historical':204},1000,{'Selfplay':500,'Historical':200})
        self.assertTrue(a['ready']);self.assertEqual(a['seconds'],20);self.assertEqual(a['terminal_per_second'],{'Selfplay':.3,'Historical':.2})
    def test_long_history_never_blocks_native_rate(self):
        r=CounterRates();a=r.observe(3600,{'all':7200});self.assertEqual(a['terminal_per_second']['all'],2)
        self.assertEqual(a['timestamp_basis'],'native-counters-session-average')
        a=r.observe(3605,{'all':7215});self.assertEqual(a['terminal_per_second']['all'],3)
        self.assertEqual(a['seconds'],5)
    def test_observed_window_is_bounded_and_counter_stall_is_zero(self):
        r=CounterRates()
        for t in range(0,2000,5):a=r.observe(t,{'all':t})
        self.assertEqual(a['seconds'],600);self.assertEqual(a['terminal_per_second']['all'],1)
        r=CounterRates();r.observe(1000,{'all':100});a=r.observe(1005,{'all':100});self.assertEqual(a['terminal_per_second']['all'],0)
    def test_missing_counter_is_not_fabricated(self):
        self.assertFalse(CounterRates().observe(None,{})['ready'])

    def test_loading_zeros_and_inherited_counts_are_excluded(self):
        r=CounterRates()
        def read(t,n):return r.observe(t,{'all':n},1000,{'all':500},exclude_loading=True)
        for t in (1000,1060,1120):self.assertFalse(read(t,500)['ready'])
        self.assertFalse(read(1125,505)['ready'])
        a=read(1130,515);self.assertEqual(a['seconds'],5);self.assertEqual(a['terminal_per_second']['all'],2)
        self.assertEqual(read(1135,515)['terminal_per_second']['all'],1)

    def test_native_loading_end_counts_first_completed_games(self):
        r=CounterRates()
        def read(t,n):return r.observe(t,{'all':n},1000,{'all':500},exclude_loading=True,active_start=1120)
        self.assertFalse(read(1100,500)['ready'])
        self.assertFalse(read(1120,500)['ready'])
        a=read(1125,525);self.assertEqual(a['seconds'],5);self.assertEqual(a['terminal_per_second']['all'],5)
        a=read(1130,550);self.assertEqual(a['terminal_per_second']['all'],5)

    def test_late_open_waits_for_two_actual_observations(self):
        r=CounterRates();self.assertFalse(r.observe(3600,{'all':7200},exclude_loading=True)['ready'])
        a=r.observe(3605,{'all':7215},exclude_loading=True)
        self.assertEqual(a['terminal_per_second']['all'],3)
        a=r.observe(4210,{'all':7215},exclude_loading=True)
        # No fabricated positive estimate over an unobserved startup interval.
        self.assertFalse(a['ready'])

    def test_new_clock_before_second_sample_discards_old_loading_anchor(self):
        r=CounterRates()
        self.assertFalse(r.observe(300,{'all':10},exclude_loading=True)['ready'])
        self.assertFalse(r.observe(20,{'all':1},exclude_loading=True)['ready'])
        a=r.observe(22,{'all':5},exclude_loading=True)
        self.assertEqual(a['terminal_per_second']['all'],2)
