from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from PIL import Image,ImageDraw
from desktop.jev_sidecar.table_counters import TableCounterReader,parse_counter_row,parse_single_counter


def row(sticks,honba):
    return [[[[210,0],[300,0],[300,35],[210,35]],f'1×{honba}',.95],
            [[[15,0],[90,0],[90,35],[15,35]],f'×{sticks}',.96]]


def patterned_image():
    image=Image.new('RGB',(400,80),'#18364b');draw=ImageDraw.Draw(image)
    draw.rectangle((25,20,70,60),fill='white')
    return image


class TableCounterTests(unittest.TestCase):
    def test_sticks_left_honba_right_even_when_ocr_order_is_reversed(self):
        self.assertEqual(parse_counter_row(row(2,3)),{'honba':3,'sticks':2})
        self.assertEqual(parse_counter_row(row(12,0)),{'honba':0,'sticks':12})
        self.assertEqual(parse_single_counter([['x O',.9]]),0)
        self.assertIsNone(parse_single_counter([['999',.9]]))

    def test_actual_read_zero_reset_and_missing_frame_are_silent(self):
        reader=TableCounterReader();ocr=Mock(return_value=(row(2,3),None))
        image=patterned_image();region={'rect':[0,0,1,1]}
        self.assertEqual(reader.read(image,region,lambda:ocr),{'honba':3,'sticks':2})
        ocr.return_value=(row(0,0),None)
        self.assertEqual(reader.read(image,region,lambda:ocr),{'honba':0,'sticks':0})
        ocr.side_effect=RuntimeError('no text this frame')
        self.assertEqual(reader.read(image,region,lambda:ocr),{'honba':0,'sticks':0})
        first=TableCounterReader()
        self.assertEqual(first.read(Image.new('RGB',(400,80)),region,lambda:ocr),{'honba':None,'sticks':None})

    def test_recognition_only_fallback_reads_each_number_not_the_stick_icon(self):
        reader=TableCounterReader()
        ocr=Mock(side_effect=[([],None),([['x12',.95]],None),([['3',.94]],None)])
        self.assertEqual(reader.read(patterned_image(),{'rect':[0,0,1,1]},lambda:ocr),{'honba':3,'sticks':12})

    def test_screen_reader_exposes_counters_with_its_existing_ocr_engine(self):
        from desktop.jev_sidecar.screen_state import ScreenStateReader
        from desktop.jev_sidecar.tiles import parse
        from desktop.jev_sidecar.vision import Detection
        reader=ScreenStateReader();reader.ocr=Mock(return_value=(row(4,2),None))
        found={'hand':[Detection(t,1,(0,0,1,1)) for t in parse('123456m345p67s115z')]}
        with patch.object(reader,'buttons',return_value=[]):
            result=reader.read(patterned_image(),{'counters':{'rect':[0,0,1,1]}},None,found)
        self.assertEqual(result['counters'],{'honba':2,'sticks':4})

    def test_widget_update_invalidates_previous_recommendation_and_handles_zero(self):
        from PySide6.QtWidgets import QApplication,QSpinBox
        from desktop.jev_sidecar.app import Sidecar
        app=QApplication.instance() or QApplication([])
        panel=SimpleNamespace(honba=QSpinBox(),sticks=QSpinBox(),invalidate=Mock())
        panel.honba.setRange(0,99);panel.sticks.setRange(0,99)
        Sidecar.update_counters(panel,{'honba':3,'sticks':2})
        self.assertEqual((panel.honba.value(),panel.sticks.value()),(3,2))
        panel.invalidate.assert_called_once()
        Sidecar.update_counters(panel,{'honba':0,'sticks':0})
        self.assertEqual((panel.honba.value(),panel.sticks.value()),(0,0))
        self.assertEqual(panel.invalidate.call_count,2)
        Sidecar.update_counters(panel,{'honba':None,'sticks':None})
        self.assertEqual(panel.invalidate.call_count,2)


if __name__=='__main__':unittest.main()
