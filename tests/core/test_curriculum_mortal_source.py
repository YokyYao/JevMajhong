import unittest
from tools.curriculum_mortal_source import public_events


class MortalPublicInputTests(unittest.TestCase):
    def test_other_initial_hand_or_draw_fails_closed(self):
        for event in [dict(type='start_kyoku',tehais=[['1m'],['?'],['?'],['?']]),
                      dict(type='tsumo',actor=0,pai='1m')]:
            with self.assertRaises(ValueError):public_events([event],1)

    def test_own_hand_and_own_draw_are_preserved_without_mutating_source(self):
        event=dict(type='start_kyoku',tehais=[['?'],['1m'],['?'],['?']])
        self.assertEqual(public_events([event,dict(type='tsumo',actor=1,pai='2m')],1)[0],event)
        start=dict(type='start_game',names=['a','b','c','d'])
        projected=public_events([start],1)[0]
        self.assertEqual(projected['id'],1)
        self.assertEqual(start['names'],['a','b','c','d'])


if __name__=='__main__':unittest.main()
