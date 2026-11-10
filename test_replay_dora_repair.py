import unittest

from replay_dora_repair import DoraChronology


def start():
    return {'type':'start_kyoku','bakaze':'E','oya':0,'honba':0,'dora_marker':'1m'}


class DoraRepairTests(unittest.TestCase):
    def test_future_kan_reveal_does_not_exist_at_initial_prefix(self):
        events=[start(),{'type':'dahai'},
            {'type':'ankan','actor':1,'consumed':['2p']*4},
            {'type':'dora','dora_marker':'3s'},{'type':'dahai'}]
        timeline=DoraChronology(events)
        initial=(0,0,0,0,((),(),(),()))
        self.assertEqual(timeline.prefixes[initial],{('1m',)})
        after_kan=(0,0,0,1,((),(('Ankan',('2p',)*4),),(),()))
        self.assertEqual(timeline.prefixes[after_kan],{('1m',),('1m','3s')})
        after_discard=(0,0,0,2,after_kan[-1])
        self.assertEqual(timeline.prefixes[after_discard],{('1m','3s')})

    def test_equal_named_indicators_are_not_deduplicated(self):
        events=[start(),{'type':'dahai'},
            {'type':'ankan','actor':1,'consumed':['2p']*4},
            {'type':'dora','dora_marker':'1m'},{'type':'dahai'}]
        timeline=DoraChronology(events)
        key=(0,0,0,2,((),(('Ankan',('2p',)*4),),(),()))
        self.assertEqual(timeline.prefixes[key],{('1m','1m')})

    def test_kakan_replaces_pon_in_public_prefix(self):
        events=[start(),{'type':'dahai'},
            {'type':'pon','actor':2,'pai':'5p','consumed':['5pr','5p']},
            {'type':'dahai'}, {'type':'kakan','actor':2,'pai':'5p'},
            {'type':'dora','dora_marker':'C'},{'type':'dahai'}]
        timeline=DoraChronology(events)
        key=(0,0,0,3,((),(),(('Kakan',('0p','5p','5p','5p')),),()))
        self.assertEqual(timeline.prefixes[key],{('1m','7z')})


if __name__=='__main__':unittest.main()
