import unittest

from desktop.jev_sidecar.autoplay import ClickPlan
from desktop.jev_sidecar.match_series import MatchPage, MatchSeries


class MatchSeriesTests(unittest.TestCase):
    def observe_twice(self, series, kind=None, playing=False):
        page = MatchPage(kind, ClickPlan(.5, .5, kind)) if kind else None
        self.assertIsNone(series.observe(page, playing))
        return series.observe(page, playing), page

    def test_resuming_from_results_counts_only_new_matches(self):
        series = MatchSeries()
        series.start(1)
        plan, page = self.observe_twice(series, 'results')
        self.assertEqual(plan, page.action)
        self.assertEqual(series.completed, 0)
        self.observe_twice(series, playing=True)
        plan, _ = self.observe_twice(series, 'results')
        self.assertIsNone(plan)
        self.assertEqual(series.completed, 1)
        self.assertEqual(series.phase, 'complete')
        plan, _ = self.observe_twice(series, 'rewards')
        self.assertIsNone(plan)

    def test_each_transition_requires_a_stable_page_and_only_clicks_once(self):
        series = MatchSeries()
        series.start(2)
        self.observe_twice(series, playing=True)
        for kind in ('results', 'rank', 'rewards', 'confirm'):
            plan, page = self.observe_twice(series, kind)
            self.assertEqual(plan, page.action)
            series.clicked(page)
            self.assertIsNone(series.observe(page))
            self.assertEqual(series.completed, 1)
        self.assertEqual(series.phase, 'queued')
        self.observe_twice(series, playing=True)
        plan, _ = self.observe_twice(series, 'results')
        self.assertIsNone(plan)
        self.assertEqual(series.completed, 2)

    def test_chest_page_between_rank_and_rewards_gets_its_own_click(self):
        series = MatchSeries()
        series.start(2)
        self.observe_twice(series, playing=True)
        for kind in ('results', 'rank', 'chest', 'rewards', 'confirm'):
            plan, page = self.observe_twice(series, kind)
            self.assertEqual(plan, page.action)
            series.clicked(page)
            self.assertIsNone(series.observe(page))
            self.assertEqual(series.completed, 1)
        self.assertEqual(series.phase, 'queued')

    def test_unrecognized_screen_and_stopped_series_do_not_click(self):
        series = MatchSeries()
        series.start(3)
        self.observe_twice(series, playing=True)
        self.assertIsNone(series.observe())
        self.assertIsNone(series.observe())
        self.assertEqual(series.completed, 0)
        series.stop()
        plan, _ = self.observe_twice(series, 'results')
        self.assertIsNone(plan)
        self.assertEqual(series.completed, 0)

    def test_can_act_on_first_visible_confirmation_without_prior_button_clicks(self):
        series = MatchSeries()
        series.start(2)
        self.observe_twice(series, playing=True)
        plan, page = self.observe_twice(series, 'confirm')
        self.assertEqual(plan, page.action)
        self.assertEqual(series.completed, 1)
        series.clicked(page)
        self.assertIsNone(series.observe(page))

    def test_resumes_at_confirmation_after_restart_and_enforces_limit(self):
        series = MatchSeries()
        plan, _ = self.observe_twice(series, 'confirm')
        self.assertIsNone(plan)
        series.start(1)
        plan, page = self.observe_twice(series, 'confirm')
        self.assertEqual(plan, page.action)
        series.clicked(page)
        self.assertEqual(series.phase, 'queued')
        self.assertEqual(series.completed, 0)
        self.assertIsNone(series.observe(page))
        self.observe_twice(series, playing=True)
        plan, _ = self.observe_twice(series, 'confirm')
        self.assertIsNone(plan)
        self.assertEqual(series.completed, 1)
        self.assertEqual(series.phase, 'complete')

    def test_current_page_determines_action_even_when_pages_are_out_of_order(self):
        series = MatchSeries()
        series.start(2)
        self.observe_twice(series, playing=True)
        for kind in ('rank', 'results', 'confirm'):
            action, page = self.observe_twice(series, kind)
            self.assertEqual(action, page.action)
            series.clicked(page)
        self.assertEqual(series.completed, 1)

    def test_limit_is_enforced_when_first_visible_page_is_confirmation(self):
        series = MatchSeries()
        series.start(1)
        self.observe_twice(series, playing=True)
        action, _ = self.observe_twice(series, 'confirm')
        self.assertIsNone(action)
        self.assertEqual(series.completed, 1)
        self.assertEqual(series.phase, 'complete')

    def test_same_confirmation_gets_only_one_delayed_retry(self):
        series = MatchSeries()
        series.start(2)
        self.observe_twice(series, playing=True)
        action, page = self.observe_twice(series, 'confirm')
        self.assertIsNotNone(action)
        series.clicked(page, now=5)
        self.assertIsNone(series.observe(page, now=9))
        self.assertEqual(series.observe(page, now=10), page.action)
        series.clicked(page, now=10)
        self.assertIsNone(series.observe(page, now=15))
        self.assertEqual(series.completed, 1)

    def test_manual_matching_can_start_new_play_without_automatic_confirmation(self):
        series = MatchSeries()
        series.start(2)
        self.observe_twice(series, playing=True)
        self.observe_twice(series, 'rank')
        self.assertEqual(series.completed, 1)
        self.observe_twice(series, playing=True)
        self.assertEqual(series.phase, 'playing')
        action, _ = self.observe_twice(series, 'confirm')
        self.assertIsNone(action)
        self.assertEqual(series.completed, 2)
