"""Boundary and denominator checks for the acquired-corpus analysis."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_team_corpus as a

class TestTeamCorpus(unittest.TestCase):
    def test_apostrophes_and_hyphens(self):
        self.assertEqual(a.tokens("I don’t need a lie-in; I'm fine."), ['i',"don't",'need','a','lie','in',"i'm",'fine'])

    def test_overlap_repetition_denominator(self):
        self.assertEqual(a.repeated_fraction(['a','b','a','b','a'],3),1/3)
        self.assertEqual(a.repeated_fraction(['a','a','a','a'],3),.5)
        self.assertIsNone(a.repeated_fraction(['a'],3))

    def test_one_sentence_overlap_is_missing_not_zero(self):
        self.assertIsNone(a.features('One long sentence without punctuation')['adjacent_jaccard'])
        self.assertEqual(a.features('One. Two.')['adjacent_jaccard'],0)

    def test_prefix_exact_length_and_punctuation(self):
        text='One two. Three four! Five six.'
        self.assertEqual(a.prefix(text,4),'One two. Three four')
        self.assertEqual(a.features(a.prefix(text,4))['sentences'],2)

    def test_mattr_rolling_window(self):
        text=' '.join(['one']*50+['two'])
        self.assertAlmostEqual(a.features(text)['mattr50'],.03)

    def test_exact_inference(self):
        x=a.paired_summary([1]*6)
        self.assertEqual(x['ci95_question_bootstrap'],[1,1])
        self.assertEqual(a.signflip_p([1]*6),2/64)
        self.assertEqual(a.signflip_p([0]*6),1)
        self.assertEqual(a.paired_summary([])['n_pairs'],0)
        self.assertEqual(a.paired_summary([1,2])['n_pairs'],2)

    def test_rank_ties(self):
        self.assertAlmostEqual(a.spearman([1,2,3],[4,4,8]),.8660254037844387)
        self.assertIsNone(a.spearman([1,2,3],[4,4,4]))

if __name__=='__main__': unittest.main()
