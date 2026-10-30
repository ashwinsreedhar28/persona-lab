import math
import unittest
from evaluation.local_pilot import paired,score

class PilotChecks(unittest.TestCase):
    def test_arithmetic_scores_the_final_number(self):
        row={'kind':'arithmetic','target':777}
        self.assertTrue(score(row,'Intermediate: 3. Final result: 777'))
        self.assertFalse(score(row,'777, but the final result is 2'))
        self.assertFalse(score(row,'No number.'))
    def test_structured_utility_requires_correct_content(self):
        self.assertTrue(score({'kind':'sort','target':[1,2,3]},'[1, 2, 3]'))
        self.assertFalse(score({'kind':'sort','target':[1,2,3]},'[3, 2, 1]'))
        self.assertFalse(score({'kind':'sort','target':[1,2,3]},'Here: [1,2,3]'))
        self.assertTrue(score({'kind':'extract','target':'abc'},'"abc"'))
        self.assertFalse(score({'kind':'uppercase','target':'ABC'},'abc'))
    def test_paired_counts_keep_both_directions(self):
        result=paired([True,True,False],[False,True,True])
        self.assertEqual((result['n'],result['base'],result['edited']),(3,2,2))
        self.assertEqual(result['delta_percentage_points'],0)
        self.assertEqual(result['base_yes_edited_no'],1)
        self.assertEqual(result['base_no_edited_yes'],1)
        self.assertAlmostEqual(result['paired_standard_error_pp'],100/math.sqrt(3))
        self.assertIsNone(paired([],[])['delta_percentage_points'])
        with self.assertRaises(ValueError):paired([True],[])
if __name__=='__main__':unittest.main()
