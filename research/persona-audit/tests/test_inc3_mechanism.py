import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_inc3_mechanism import measure

class TestDecomposition(unittest.TestCase):
    def test_repeated_single_function_word(self):
        r=measure(['the']*50,{'the'})
        self.assertEqual(r['mattr50'],.02)
        self.assertEqual(r['function_repeat_loss'],.98)
        self.assertEqual(r['nonlist_repeat_loss'],0)
        self.assertIsNone(r['nonlist_mattr25'])
    def test_equal_repetition_of_two_classes(self):
        r=measure(['the','cat']*25,{'the'})
        self.assertEqual(r['mattr50'],.04)
        self.assertEqual(r['function_repeat_loss'],.48)
        self.assertEqual(r['nonlist_repeat_loss'],.48)
        self.assertEqual(r['nonlist_mattr25'],.04)
    def test_all_unique(self):
        ts=[str(i) for i in range(70)]
        r=measure(ts,set(ts[:20]))
        self.assertEqual(r['mattr50'],1)
        self.assertEqual(r['function_repeat_loss'],0)
        self.assertEqual(r['nonlist_repeat_loss'],0)
        self.assertEqual(r['nonlist_mattr25'],1)
    def test_identity_after_lexicon_change(self):
        ts=['the','cat','and','a','dog']*15
        a=measure(ts,{'the','a'});b=measure(ts,{'the','a','and'})
        self.assertEqual(a['mattr50'],b['mattr50'])
        self.assertAlmostEqual(a['function_repeat_loss']+a['nonlist_repeat_loss'],1-a['mattr50'])
        self.assertGreater(b['function_repeat_loss'],a['function_repeat_loss'])
if __name__=='__main__':unittest.main()
