import copy
import unittest
from tools import curriculum_call_type_balance_v1 as balance


class CallTypeBalanceTests(unittest.TestCase):
    def test_source_mass_exact_and_absent_class_not_created(self):
        masses=balance.capped_equal_mass({'ankan':10,'other':90},.2,100)
        self.assertEqual(masses,{'ankan':.1,'other':.1})
        self.assertNotIn('daiminkan',masses)
        self.assertEqual(sum(masses.values()),.2)

    def test_cap_redistributes_excess_without_dropping_the_rare_class(self):
        masses=balance.capped_equal_mass({'kakan':1,'other':999},.2,1000)
        self.assertAlmostEqual(masses['kakan'],.016)
        self.assertAlmostEqual(masses['other'],.184)
        self.assertLessEqual(masses['kakan']*1000,16)
        with self.assertRaises(ValueError):balance.capped_equal_mass({'only':1},.5,1000)

    def test_raw_rows_and_other_sources_unchanged(self):
        rows=[]
        for source in balance.MASS:
            operations=('chi','chi','pon','none') if source.endswith(':call') else (
                ('ankan','daiminkan','kakan','none') if source.endswith(':kan') else ('dahai',)*4)
            for op in operations:
                rows.append(dict(state={'unchanged':True},_meta={'id':str(len(rows))},
                    questions={'action':dict(src='curriculum2:'+source,label='a',criteria={'a':{'type':op}})}))
        before=copy.deepcopy(rows);entries,summary=balance.build(rows)
        self.assertEqual(rows,before)
        self.assertEqual(len(entries),len(rows))
        for source,data in summary['sources'].items():
            self.assertAlmostEqual(sum(data['class_mass'].values()),balance.MASS[source])
            if source not in balance.SOURCES:
                self.assertEqual(data['class_row_weights']['unchanged_source'],data['previous_uniform_row_weight'])
        self.assertLessEqual(summary['max_row_weight'],16+1e-12)
        with self.assertRaises(ValueError):balance.build(rows+[rows[0]])
        rows[0]['questions']['action']['target']={'a':1}
        with self.assertRaises(ValueError):balance.build(rows)


if __name__=='__main__':unittest.main()
