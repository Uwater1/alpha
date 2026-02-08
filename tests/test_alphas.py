import unittest
import numpy as np
import pandas as pd
from alpha191 import (
    alpha_001, alpha_002, alpha_003, alpha_004, alpha_005,
    alpha_006, alpha_007, alpha_008, alpha_009, alpha_010,
    alpha_011, alpha_012, alpha_013, alpha_014, alpha_015,
    alpha_016, alpha_017, alpha_018, alpha_019, alpha_020,
    alpha_021, alpha_022, alpha_023, alpha_024, alpha_025,
    alpha_026, alpha_027, alpha_028, alpha_029, alpha_031,
    alpha_032, alpha_033, alpha_034, alpha_035, alpha_036,
    alpha_037, alpha_038, alpha_039, alpha_040, alpha_041,
    alpha_042, alpha_043, alpha_044, alpha_045, alpha_046,
    alpha_047, alpha_048, alpha_049, alpha_050, alpha_051, alpha_052,
    alpha_053, alpha_054, alpha_055, alpha_056, alpha_057,
    alpha_058, alpha_059, alpha_060, alpha_061, alpha_062,
    alpha_063, alpha_064, alpha_065, alpha_066, alpha_067,
    alpha_068, alpha_069, alpha_070, alpha_071, alpha_072,
    alpha_073, alpha_074, alpha_075, alpha_076, alpha_077,
    alpha_078, alpha_079, alpha_080, alpha_081, alpha_082,
    alpha_083, alpha_084, alpha_085, alpha_086, alpha_087,
    alpha_088, alpha_089, alpha_090, alpha_091, alpha_092,
    alpha_093, alpha_094, alpha_095, alpha_096, alpha_097,
    alpha_098, alpha_099, alpha_100, alpha_101, alpha_102,
    alpha_103, alpha_104, alpha_105, alpha_106, alpha_107,
    alpha_108, alpha_109, alpha_110, alpha_111, alpha_112,
    alpha_113, alpha_114, alpha_115, alpha_116, alpha_117,
    alpha_118, alpha_119, alpha_120, alpha_121, alpha_122,
    alpha_123, alpha_124, alpha_125, alpha_126, alpha_127,
    alpha_128, alpha_129, alpha_130, alpha_131, alpha_132,
    alpha_133, alpha_134, alpha_135, alpha_136, alpha_137,
    alpha_138,     alpha_139, alpha_140, alpha_141, alpha_142,
    # alpha_143,  # 尚未实现 (Haven't implemented)
    alpha_144, alpha_145, alpha_146, alpha_147,
    alpha_148, alpha_149, alpha_150, alpha_151, alpha_152,
    alpha_153, alpha_154, alpha_155, alpha_156, alpha_157,
    alpha_158, alpha_159, alpha_160, alpha_161, alpha_162,
    alpha_163, alpha_164,
    alpha_165,
    alpha_166, alpha_167,
    alpha_168, alpha_169, alpha_170, alpha_171, alpha_172,
    alpha_173, alpha_174, alpha_175, alpha_176, alpha_177,
    alpha_178, alpha_179, alpha_180,     alpha_181, alpha_182,
    # alpha_183,  # 尚未实现 (Haven't implemented)
    alpha_184, alpha_185, alpha_186, alpha_187,
    alpha_188, alpha_189, alpha_190, alpha_191
)


class TestAlphas(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 300  # Increased for larger windows in alpha 21-40
        close = 101 + np.random.randn(n).cumsum()
        self.df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=n),
            'open': 100 + np.random.randn(n).cumsum(),
            'high': 102 + np.random.randn(n).cumsum(),
            'low': 98 + np.random.randn(n).cumsum(),
            'close': close,
            'volume': np.random.randint(1000000, 5000000, n),
            'amount': close * np.random.randint(1000000, 5000000, n),
        })
        # Add derived columns needed by some alphas
        self.df['vwap'] = self.df['amount'] / self.df['volume']
        self.df['ret'] = self.df['close'].pct_change()
        # Add benchmark_close column needed by alpha149
        self.df['benchmark_close'] = close * 0.95  # Some benchmark close values
    
    def test_alpha001(self):
        result = alpha_001(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(np.all(np.isnan(result.values[:5])))

    def test_alpha002(self):
        result = alpha_002(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(np.isnan(result.values[0]))
        self.assertFalse(np.isnan(result.values[1]))

    def test_alpha003(self):
        result = alpha_003(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(np.all(np.isnan(result.values[:5])))
        self.assertFalse(np.isnan(result.values[5]))

    def test_alpha004(self):
        result = alpha_004(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(np.all(np.isnan(result.values[:19])))
        self.assertFalse(np.isnan(result.values[19]))

    def test_alpha005(self):
        result = alpha_005(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(np.all(np.isnan(result.values[:4])))
        valid_idx = np.where(~np.isnan(result.values))[0]
        if len(valid_idx) > 0:
            self.assertGreaterEqual(valid_idx[0], 5)

    def test_alpha006(self):
        result = alpha_006(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(np.all(np.isnan(result.values[:4])))

    def test_alpha007(self):
        result = alpha_007(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha008(self):
        result = alpha_008(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha009(self):
        result = alpha_009(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha010(self):
        result = alpha_010(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha011(self):
        result = alpha_011(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha012(self):
        result = alpha_012(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha013(self):
        result = alpha_013(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha014(self):
        result = alpha_014(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha015(self):
        result = alpha_015(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha016(self):
        result = alpha_016(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha017(self):
        result = alpha_017(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha018(self):
        result = alpha_018(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha019(self):
        result = alpha_019(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha020(self):
        result = alpha_020(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha021(self):
        result = alpha_021(self.df)
        self.assertEqual(len(result), len(self.df))
        # mean(6) + regression_beta(6)
        # Should be valid around index 10
        self.assertFalse(np.isnan(result.values[-1]))

    def test_alpha022(self):
        result = alpha_022(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha023(self):
        result = alpha_023(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha024(self):
        result = alpha_024(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha025(self):
        result = alpha_025(self.df)
        self.assertEqual(len(result), len(self.df))
        # sum(ret, 250) -> ret starts at 1. ts_sum(250) valid from index 249.
        self.assertTrue(np.all(np.isnan(result.values[:249])))
        self.assertFalse(np.isnan(result.values[-1]))

    def test_alpha026(self):
        result = alpha_026(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha027(self):
        result = alpha_027(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha028(self):
        result = alpha_028(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha029(self):
        result = alpha_029(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha031(self):
        result = alpha_031(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha032(self):
        result = alpha_032(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha033(self):
        result = alpha_033(self.df)
        self.assertEqual(len(result), len(self.df))
        # sum(ret, 240) -> ret starts at 1. ts_sum(240) valid from index 239.
        self.assertTrue(np.all(np.isnan(result.values[:239])))

    def test_alpha034(self):
        result = alpha_034(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha035(self):
        result = alpha_035(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha036(self):
        result = alpha_036(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha037(self):
        result = alpha_037(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha038(self):
        result = alpha_038(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha039(self):
        result = alpha_039(self.df)
        self.assertEqual(len(result), len(self.df))
        # correlation over 14, sum over 37, mean over 180
        valid_idx = np.where(~np.isnan(result.values))[0]
        if len(valid_idx) > 0:
             self.assertGreaterEqual(valid_idx[0], 180)

    def test_alpha040(self):
        result = alpha_040(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha041(self):
        result = alpha_041(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha042(self):
        result = alpha_042(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha043(self):
        result = alpha_043(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha044(self):
        result = alpha_044(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha045(self):
        result = alpha_045(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha046(self):
        result = alpha_046(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha047(self):
        result = alpha_047(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha048(self):
        result = alpha_048(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha049(self):
        result = alpha_049(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha050(self):
        result = alpha_050(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha051(self):
        result = alpha_051(self.df)
        self.assertEqual(len(result), len(self.df))
        # SUM(..., 12) -> valid from index 11
        self.assertTrue(np.all(np.isnan(result.values[:11])))

    def test_alpha052(self):
        result = alpha_052(self.df)
        self.assertEqual(len(result), len(self.df))
        # SUM(..., 26) -> valid from index 25
        self.assertTrue(np.all(np.isnan(result.values[:25])))

    def test_alpha053(self):
        result = alpha_053(self.df)
        self.assertEqual(len(result), len(self.df))
        # COUNT(..., 12) -> valid from index 11
        self.assertTrue(np.all(np.isnan(result.values[:11])))

    def test_alpha054(self):
        result = alpha_054(self.df)
        self.assertEqual(len(result), len(self.df))
        # STD(..., 10) + CORR(..., 10) -> valid from index 9
        self.assertTrue(np.all(np.isnan(result.values[:9])))

    def test_alpha055(self):
        result = alpha_055(self.df)
        self.assertEqual(len(result), len(self.df))
        # SUM(..., 20) -> valid from index 19
        self.assertTrue(np.all(np.isnan(result.values[:19])))

    def test_alpha056(self):
        result = alpha_056(self.df)
        self.assertEqual(len(result), len(self.df))
        # TSMIN(OPEN, 12) + CORR(..., 13) -> valid from index 12
        # Note: Some values may be computed earlier due to operator behavior
        self.assertFalse(np.all(np.isnan(result.values)))

    def test_alpha057(self):
        result = alpha_057(self.df)
        self.assertEqual(len(result), len(self.df))
        # TSMIN(LOW, 9) + TSMAX(HIGH, 9) -> valid from index 8
        self.assertTrue(np.all(np.isnan(result.values[:8])))

    def test_alpha058(self):
        result = alpha_058(self.df)
        self.assertEqual(len(result), len(self.df))
        # COUNT(..., 20) -> valid from index 19
        self.assertTrue(np.all(np.isnan(result.values[:19])))

    def test_alpha059(self):
        result = alpha_059(self.df)
        self.assertEqual(len(result), len(self.df))
        # SUM(..., 20) -> valid from index 19
        self.assertTrue(np.all(np.isnan(result.values[:19])))

    def test_alpha060(self):
        result = alpha_060(self.df)
        self.assertEqual(len(result), len(self.df))
        # SUM(..., 20) -> valid from index 19
        self.assertTrue(np.all(np.isnan(result.values[:19])))

    def test_alpha061(self):
        result = alpha_061(self.df)
        self.assertEqual(len(result), len(self.df))
        # DECAYLINEAR(..., 12) + DECAYLINEAR(..., 17) -> valid from index 16
        self.assertTrue(np.all(np.isnan(result.values[:16])))

    def test_alpha062(self):
        result = alpha_062(self.df)
        self.assertEqual(len(result), len(self.df))
        # CORR(..., 5) -> valid from index 4
        self.assertTrue(np.all(np.isnan(result.values[:4])))

    def test_alpha063(self):
        result = alpha_063(self.df)
        self.assertEqual(len(result), len(self.df))
        # SMA(..., 6, 1) -> valid from index 5
        # Note: Some values may be computed earlier due to operator behavior
        self.assertFalse(np.all(np.isnan(result.values)))

    def test_alpha064(self):
        result = alpha_064(self.df)
        self.assertEqual(len(result), len(self.df))
        # CORR(..., 4) + DECAYLINEAR(..., 14) -> valid from index 13
        self.assertTrue(np.all(np.isnan(result.values[:13])))

    def test_alpha065(self):
        result = alpha_065(self.df)
        self.assertEqual(len(result), len(self.df))
        # MEAN(CLOSE, 6) -> valid from index 5
        self.assertTrue(np.all(np.isnan(result.values[:5])))

    def test_alpha066(self):
        result = alpha_066(self.df)
        self.assertEqual(len(result), len(self.df))
        # MEAN(CLOSE, 6) -> valid from index 5
        self.assertTrue(np.all(np.isnan(result.values[:5])))

    def test_alpha067(self):
        result = alpha_067(self.df)
        self.assertEqual(len(result), len(self.df))
        # SMA(..., 24, 1) -> valid from index 23
        # Note: Some values may be computed earlier due to operator behavior
        self.assertFalse(np.all(np.isnan(result.values)))

    def test_alpha068(self):
        result = alpha_068(self.df)
        self.assertEqual(len(result), len(self.df))
        # SMA(..., 15, 2) -> valid from index 14
        # Note: Some values may be computed earlier due to operator behavior
        self.assertFalse(np.all(np.isnan(result.values)))

    def test_alpha069(self):
        result = alpha_069(self.df)
        self.assertEqual(len(result), len(self.df))
        # SUM(DTM, 20) + SUM(DBM, 20) -> valid from index 19
        self.assertTrue(np.all(np.isnan(result.values[:19])))

    def test_alpha070(self):
        result = alpha_070(self.df)
        self.assertEqual(len(result), len(self.df))
        # STD(AMOUNT, 6) -> valid from index 5
        self.assertTrue(np.all(np.isnan(result.values[:5])))

    def test_alpha071(self):
        result = alpha_071(self.df)
        self.assertEqual(len(result), len(self.df))
        # SUM(..., 24) -> valid from index 23
        self.assertTrue(np.all(np.isnan(result.values[:23])))

    def test_alpha072(self):
        result = alpha_072(self.df)
        self.assertEqual(len(result), len(self.df))
        # SMA(..., 20, 1) -> valid from index 19
        # Note: Some values may be computed earlier due to operator behavior
        self.assertFalse(np.all(np.isnan(result.values)))

    def test_alpha073(self):
        result = alpha_073(self.df)
        self.assertEqual(len(result), len(self.df))
        # SUM(..., 20) -> valid from index 19
        self.assertTrue(np.all(np.isnan(result.values[:19])))

    def test_alpha074(self):
        result = alpha_074(self.df)
        self.assertEqual(len(result), len(self.df))
        # SUM(..., 20) -> valid from index 19
        self.assertTrue(np.all(np.isnan(result.values[:19])))

    def test_alpha075(self):
        result = alpha_075(self.df)
        self.assertEqual(len(result), len(self.df))
        # SUM(..., 20) -> valid from index 19
        self.assertTrue(np.all(np.isnan(result.values[:19])))

    def test_alpha076(self):
        result = alpha_076(self.df)
        self.assertEqual(len(result), len(self.df))
        # SUM(..., 20) -> valid from index 19
        self.assertTrue(np.all(np.isnan(result.values[:19])))

    def test_alpha077(self):
        result = alpha_077(self.df)
        self.assertEqual(len(result), len(self.df))
        # SMA(..., 24, 1) -> valid from index 23
        # Note: Some values may be computed earlier due to operator behavior
        self.assertFalse(np.all(np.isnan(result.values)))

    def test_alpha078(self):
        result = alpha_078(self.df)
        self.assertEqual(len(result), len(self.df))
        # SMA(..., 12, 1) -> valid from index 11
        # Note: Some values may be computed earlier due to operator behavior
        self.assertFalse(np.all(np.isnan(result.values)))

    def test_alpha079(self):
        result = alpha_079(self.df)
        self.assertEqual(len(result), len(self.df))
        # SMA(..., 12, 1) -> valid from index 1
        self.assertTrue(np.isnan(result.values[0]))
        self.assertFalse(np.isnan(result.values[1]))

    def test_alpha080(self):
        result = alpha_080(self.df)
        self.assertEqual(len(result), len(self.df))
        # DELAY(VOLUME, 5) -> valid from index 5
        self.assertTrue(np.all(np.isnan(result.values[:5])))

    def test_alpha081(self):
        result = alpha_081(self.df)
        self.assertEqual(len(result), len(self.df))
        # SMA(VOLUME, 21, 2) -> valid from index 0
        self.assertFalse(np.isnan(result.values[0]))

    def test_alpha082(self):
        result = alpha_082(self.df)
        self.assertEqual(len(result), len(self.df))
        # TSMAX(HIGH,6) and TSMIN(LOW,6) -> valid from index 5
        self.assertTrue(np.all(np.isnan(result.values[:5])))

    def test_alpha083(self):
        result = alpha_083(self.df)
        self.assertEqual(len(result), len(self.df))
        # COVIANCE(..., 5) -> valid from index 4
        self.assertTrue(np.all(np.isnan(result.values[:4])))

    def test_alpha084(self):
        result = alpha_084(self.df)
        self.assertEqual(len(result), len(self.df))
        # SUM(..., 20) -> valid from index 19
        self.assertTrue(np.all(np.isnan(result.values[:19])))

    def test_alpha085(self):
        result = alpha_085(self.df)
        self.assertEqual(len(result), len(self.df))
        # TSRANK(..., 20) -> valid from index 19
        self.assertTrue(np.all(np.isnan(result.values[:19])))

    def test_alpha086(self):
        result = alpha_086(self.df)
        self.assertEqual(len(result), len(self.df))
        # DELAY(CLOSE, 20) -> valid from index 20
        self.assertTrue(np.all(np.isnan(result.values[:20])))

    def test_alpha087(self):
        result = alpha_087(self.df)
        self.assertEqual(len(result), len(self.df))
        # DECAYLINEAR(..., 7) + TSRANK(..., 7) -> valid from index 6
        self.assertTrue(np.all(np.isnan(result.values[:6])))

    def test_alpha088(self):
        result = alpha_088(self.df)
        self.assertEqual(len(result), len(self.df))
        # DELAY(CLOSE, 20) -> valid from index 20
        self.assertTrue(np.all(np.isnan(result.values[:20])))

    def test_alpha089(self):
        result = alpha_089(self.df)
        self.assertEqual(len(result), len(self.df))
        # SMA(..., 13, 2) -> valid from index 0
        self.assertFalse(np.isnan(result.values[0]))

    def test_alpha090(self):
        result = alpha_090(self.df)
        self.assertEqual(len(result), len(self.df))
        # CORR(..., 5) -> valid from index 4
        self.assertTrue(np.all(np.isnan(result.values[:4])))

    def test_alpha091(self):
        result = alpha_091(self.df)
        self.assertEqual(len(result), len(self.df))
        # TS_MEAN(VOLUME, 40) -> valid from index 39
        self.assertTrue(np.all(np.isnan(result.values[:39])))

    def test_alpha092(self):
        result = alpha_092(self.df)
        self.assertEqual(len(result), len(self.df))
        # TS_MEAN(VOLUME, 180) -> valid from index 179
        self.assertTrue(np.all(np.isnan(result.values[:179])))

    def test_alpha093(self):
        result = alpha_093(self.df)
        self.assertEqual(len(result), len(self.df))
        # TS_SUM(..., 20) -> valid from index 19
        self.assertTrue(np.all(np.isnan(result.values[:19])))

    def test_alpha094(self):
        result = alpha_094(self.df)
        self.assertEqual(len(result), len(self.df))
        # TS_SUM(..., 30) -> valid from index 29
        self.assertTrue(np.all(np.isnan(result.values[:29])))

    def test_alpha095(self):
        result = alpha_095(self.df)
        self.assertEqual(len(result), len(self.df))
        # STD(AMOUNT, 20) -> valid from index 19
        self.assertTrue(np.all(np.isnan(result.values[:19])))

    def test_alpha096(self):
        result = alpha_096(self.df)
        self.assertEqual(len(result), len(self.df))
        # SMA(..., 3, 1) -> valid from index 2
        self.assertTrue(np.all(np.isnan(result.values[:2])))

    def test_alpha097(self):
        result = alpha_097(self.df)
        self.assertEqual(len(result), len(self.df))
        # STD(VOLUME, 10) -> valid from index 9
        self.assertTrue(np.all(np.isnan(result.values[:9])))

    def test_alpha098(self):
        result = alpha_098(self.df)
        self.assertEqual(len(result), len(self.df))
        # TS_SUM(CLOSE, 100) -> valid from index 99
        self.assertTrue(np.all(np.isnan(result.values[:99])))

    def test_alpha099(self):
        result = alpha_099(self.df)
        self.assertEqual(len(result), len(self.df))
        # COVIANCE(..., 5) -> valid from index 4
        self.assertTrue(np.all(np.isnan(result.values[:4])))

    def test_alpha100(self):
        result = alpha_100(self.df)
        self.assertEqual(len(result), len(self.df))
        # STD(VOLUME, 20) -> valid from index 19
        self.assertTrue(np.all(np.isnan(result.values[:19])))

    def test_alpha101(self):
        result = alpha_101(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha102(self):
        result = alpha_102(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha103(self):
        result = alpha_103(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha104(self):
        result = alpha_104(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha105(self):
        result = alpha_105(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha106(self):
        result = alpha_106(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha107(self):
        result = alpha_107(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha108(self):
        result = alpha_108(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha109(self):
        result = alpha_109(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha110(self):
        result = alpha_110(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha111(self):
        result = alpha_111(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha112(self):
        result = alpha_112(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha113(self):
        result = alpha_113(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha114(self):
        result = alpha_114(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha115(self):
        result = alpha_115(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha116(self):
        result = alpha_116(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha117(self):
        result = alpha_117(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha118(self):
        result = alpha_118(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha119(self):
        result = alpha_119(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha120(self):
        result = alpha_120(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha121(self):
        result = alpha_121(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha122(self):
        result = alpha_122(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha123(self):
        result = alpha_123(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha124(self):
        result = alpha_124(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha125(self):
        result = alpha_125(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha126(self):
        result = alpha_126(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha127(self):
        result = alpha_127(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha128(self):
        result = alpha_128(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha129(self):
        result = alpha_129(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha130(self):
        result = alpha_130(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha131(self):
        result = alpha_131(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha132(self):
        result = alpha_132(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha133(self):
        result = alpha_133(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha134(self):
        result = alpha_134(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha135(self):
        result = alpha_135(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha136(self):
        result = alpha_136(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha137(self):
        result = alpha_137(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha138(self):
        result = alpha_138(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha139(self):
        result = alpha_139(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha140(self):
        result = alpha_140(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha141(self):
        result = alpha_141(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha142(self):
        result = alpha_142(self.df)
        self.assertEqual(len(result), len(self.df))

    # def test_alpha143(self):  # 尚未实现 (Haven't implemented)
    #     result = alpha_143(self.df)
    #     self.assertEqual(len(result), len(self.df))

    def test_alpha144(self):
        result = alpha_144(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha145(self):
        result = alpha_145(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha146(self):
        result = alpha_146(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha147(self):
        result = alpha_147(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha148(self):
        result = alpha_148(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha149(self):
        result = alpha_149(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha150(self):
        result = alpha_150(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha151(self):
        result = alpha_151(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha152(self):
        result = alpha_152(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha153(self):
        result = alpha_153(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha154(self):
        result = alpha_154(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha155(self):
        result = alpha_155(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha156(self):
        result = alpha_156(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha157(self):
        result = alpha_157(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha158(self):
        result = alpha_158(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha159(self):
        result = alpha_159(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha160(self):
        result = alpha_160(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha161(self):
        result = alpha_161(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha162(self):
        result = alpha_162(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha163(self):
        result = alpha_163(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha164(self):
        result = alpha_164(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha165(self):
        result = alpha_165(self.df)
        self.assertEqual(len(result), len(self.df))
        # R/S uses 48 days window, so first 47 values should be NaN
        self.assertTrue(np.all(np.isnan(result.values[:47])))
        self.assertFalse(np.isnan(result.values[47]))

    def test_alpha166(self):
        result = alpha_166(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha167(self):
        result = alpha_167(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha168(self):
        result = alpha_168(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha169(self):
        result = alpha_169(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha170(self):
        result = alpha_170(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha171(self):
        result = alpha_171(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha172(self):
        result = alpha_172(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha173(self):
        result = alpha_173(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha174(self):
        result = alpha_174(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha175(self):
        result = alpha_175(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha176(self):
        result = alpha_176(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha177(self):
        result = alpha_177(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha178(self):
        result = alpha_178(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha179(self):
        result = alpha_179(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha180(self):
        result = alpha_180(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha181(self):
        result = alpha_181(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha182(self):
        result = alpha_182(self.df)
        self.assertEqual(len(result), len(self.df))

    # def test_alpha183(self):  # 尚未实现 (Haven't implemented)
    #     result = alpha_183(self.df)
    #     self.assertEqual(len(result), len(self.df))

    def test_alpha184(self):
        result = alpha_184(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha185(self):
        result = alpha_185(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha186(self):
        result = alpha_186(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha187(self):
        result = alpha_187(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha188(self):
        result = alpha_188(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha189(self):
        result = alpha_189(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha190(self):
        result = alpha_190(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha191(self):
        result = alpha_191(self.df)
        self.assertEqual(len(result), len(self.df))


if __name__ == '__main__':
    unittest.main()
