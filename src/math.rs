//! Portable math: the same bits on every platform.
//!
//! The platform `ln`, `exp` and `powf` can differ in the last bit between systems (they come from
//! the C library or from compiler intrinsics), which would change random choices made with them.
//! These are ports of fdlibm (the basis of most libm implementations), which only use basic
//! floating point operations: IEEE 754 makes their results exact to the bit, so they give the same
//! result everywhere. `sqrt` needs no port: IEEE 754 requires it to be correctly rounded.

/// The natural logarithm of `x`, the same on every platform: `-inf` for 0, NaN for negative `x`
/// and NaN, `inf` for `inf`.
///
/// The platform `ln` can differ in the last bit between systems, which would change the random
/// choices made with it. This is fdlibm's `__ieee754_log` (the basis of most libm
/// implementations), which only uses basic floating point operations: IEEE 754 makes their results
/// exact to the bit, so this gives the same result everywhere, within 1 ulp of the true logarithm.
pub(crate) fn log(x: f64) -> f64 {
    // fdlibm's constants, by their exact bits
    const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000); // 6.93147180369123816490e-1
    const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76); // 1.90821492927058770002e-10
    const TWO54: f64 = f64::from_bits(0x4350_0000_0000_0000); // 1.80143985094819840000e16
    const LG1: f64 = f64::from_bits(0x3fe5_5555_5555_5593); // 6.666666666666735130e-1
    const LG2: f64 = f64::from_bits(0x3fd9_9999_9997_fa04); // 3.999999999940941908e-1
    const LG3: f64 = f64::from_bits(0x3fd2_4924_9422_9359); // 2.857142874366239149e-1
    const LG4: f64 = f64::from_bits(0x3fcc_71c5_1d8e_78af); // 2.222219843214978396e-1
    const LG5: f64 = f64::from_bits(0x3fc7_4664_96cb_03de); // 1.818357216161805012e-1
    const LG6: f64 = f64::from_bits(0x3fc3_9a09_d078_c69f); // 1.531383769920937332e-1
    const LG7: f64 = f64::from_bits(0x3fc2_f112_df3e_5244); // 1.479819860511658591e-1
    // fdlibm's special cases
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x == f64::INFINITY {
        return x;
    }

    let mut x = x;
    let mut high = (x.to_bits() >> 32) as i32;
    let mut k: i32 = 0;
    if high < 0x0010_0000 {
        // subnormal: scale up
        k -= 54;
        x *= TWO54;
        high = (x.to_bits() >> 32) as i32;
    }
    k += (high >> 20) - 1023;
    high &= 0x000f_ffff;
    let i = (high + 0x95f64) & 0x10_0000;
    // normalize x or x / 2 into [sqrt(2) / 2, sqrt(2))
    let normalized_high = (high | (i ^ 0x3ff0_0000)) as u32;
    x = f64::from_bits((u64::from(normalized_high) << 32) | (x.to_bits() & 0xffff_ffff));
    k += i >> 20;
    let f = x - 1.0;
    let dk = f64::from(k);
    if (0x000f_ffff & (2 + high)) < 3 {
        // |f| < 2^-20
        if f == 0.0 {
            return dk * LN2_HI + dk * LN2_LO;
        }
        let r = f * f * (0.5 - (1.0 / 3.0) * f);
        return if k == 0 {
            f - r
        } else {
            dk * LN2_HI - ((r - dk * LN2_LO) - f)
        };
    }
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let r = t2 + t1;
    let i = (high - 0x6147a) | (0x6b851 - high);
    if i > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 {
            f - (hfsq - s * (hfsq + r))
        } else {
            dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
        }
    } else if k == 0 {
        f - s * (f - r)
    } else {
        dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
    }
}

/// `e` to the power `x`, the same on every platform, within 1 ulp of the true value.
///
/// fdlibm's `__ieee754_exp`. Overflows to infinity above about 709.78, and underflows to 0 below
/// about -745.13.
pub(crate) fn exp(x: f64) -> f64 {
    // fdlibm's constants, by their exact bits
    const O_THRESHOLD: f64 = f64::from_bits(0x4086_2e42_fefa_39ef); // 7.09782712893383973096e2
    const U_THRESHOLD: f64 = f64::from_bits(0xc087_4910_d52d_3051); // -7.45133219101941108420e2
    const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000); // 6.93147180369123816490e-1
    const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76); // 1.90821492927058770002e-10
    const INV_LN2: f64 = f64::from_bits(0x3ff7_1547_652b_82fe); // 1.44269504088896338700
    const TWO_M1000: f64 = f64::from_bits(0x0170_0000_0000_0000); // 2^-1000
    const P1: f64 = f64::from_bits(0x3fc5_5555_5555_553e); // 1.66666666666666019037e-1
    const P2: f64 = f64::from_bits(0xbf66_c16c_16be_bd93); // -2.77777777770155933842e-3
    const P3: f64 = f64::from_bits(0x3f11_566a_af25_de2c); // 6.61375632143793436117e-5
    const P4: f64 = f64::from_bits(0xbebb_bd41_c5d2_6bf1); // -1.65339022054652515390e-6
    const P5: f64 = f64::from_bits(0x3e66_3769_72be_a4d0); // 4.13813679705723846039e-8

    let high = (x.to_bits() >> 32) as u32;
    let negative = high >> 31 == 1;
    let high = high & 0x7fff_ffff;
    if high >= 0x4086_2e42 {
        // |x| >= 709.78...
        if x.is_nan() {
            return x;
        }
        if x > O_THRESHOLD {
            return f64::INFINITY;
        }
        if x < U_THRESHOLD {
            return 0.0;
        }
    }
    // argument reduction: x = k ln2 + r, |r| <= 0.5 ln2
    let (k, hi, lo, x) = if high > 0x3fd6_2e42 {
        // |x| > 0.5 ln2
        let (k, hi, lo) = if high < 0x3ff0_a2b2 {
            // and |x| < 1.5 ln2
            if negative {
                (-1, x + LN2_HI, -LN2_LO)
            } else {
                (1, x - LN2_HI, LN2_LO)
            }
        } else {
            let k = (INV_LN2 * x + if negative { -0.5 } else { 0.5 }) as i32;
            let t = f64::from(k);
            // t * LN2_HI is exact here
            (k, x - t * LN2_HI, t * LN2_LO)
        };
        (k, hi, lo, hi - lo)
    } else if high < 0x3e30_0000 {
        // |x| < 2^-28
        return 1.0 + x;
    } else {
        (0, 0.0, 0.0, x)
    };
    // x is now in the primary range
    let t = x * x;
    let c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if k == 0 {
        return 1.0 - ((x * c) / (c - 2.0) - x);
    }
    // fdlibm's order of operations, for fdlibm's results
    let y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi);
    // multiply by 2^k through the exponent bits
    if k >= -1021 {
        f64::from_bits(y.to_bits().wrapping_add((i64::from(k) << 52) as u64))
    } else {
        f64::from_bits(y.to_bits().wrapping_add((i64::from(k + 1000) << 52) as u64)) * TWO_M1000
    }
}

/// `base` to the power `exponent`, for `base >= 0`, the same on every platform.
///
/// Computed as `exp(exponent * ln(base))`, so the relative error grows with
/// `|exponent * ln(base)|`: about 1e-13 for the operators' uses, where the result is between 0
/// and a few.
pub(crate) fn pow(base: f64, exponent: f64) -> f64 {
    debug_assert!(base >= 0.0, "pow({base}, {exponent})");
    if base == 0.0 {
        return if exponent > 0.0 {
            0.0
        } else if exponent == 0.0 {
            1.0
        } else {
            f64::INFINITY
        };
    }
    if exponent == 1.0 {
        return base;
    }
    exp(exponent * log(base))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StreamRng;
    use rand::Rng;

    fn ulps(a: f64, b: f64) -> u64 {
        (a.to_bits() as i64).abs_diff(b.to_bits() as i64)
    }

    #[test]
    fn log_is_within_one_ulp_of_std() {
        let mut rng = StreamRng::seed_from_u64(0);
        let mut values = vec![
            1.0,
            0.5,
            2.0,
            0.999,
            1.0 - 1e-12,
            1.0 + 1e-12,
            1e-300,
            f64::MIN_POSITIVE,
            5e-324,
            f64::MAX,
        ];
        // (0, 1], as used by the sampler
        values.extend((0..20_000).map(|_| 1.0 - rng.unit_f64()));
        // any positive finite value, subnormals included
        values.extend(
            (0..20_000)
                .map(|_| f64::from_bits(rng.next_u64() % 0x7ff0_0000_0000_0000))
                .filter(|&x| x > 0.0),
        );
        for x in values {
            let (ours, std) = (log(x), x.ln());
            assert!(ulps(ours, std) <= 1, "log({x:e}) = {ours:e}, std {std:e}");
        }
    }

    #[test]
    fn exp_is_within_one_ulp_of_std() {
        let mut rng = StreamRng::seed_from_u64(0);
        let mut values = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.3,
            -0.3,
            1e-30,
            -1e-30,
            700.0,
            -700.0,
            709.0,
            -744.0,
            2f64.powi(-29),
        ];
        values.extend((0..20_000).map(|_| (rng.unit_f64() - 0.5) * 2.0 * 745.0));
        values.extend((0..20_000).map(|_| (rng.unit_f64() - 0.5) * 4.0));
        for x in values {
            let (ours, std) = (exp(x), x.exp());
            assert!(ulps(ours, std) <= 1, "exp({x:e}) = {ours:e}, std {std:e}");
        }
        assert_eq!(log(0.0), f64::NEG_INFINITY);
        assert_eq!(log(f64::INFINITY), f64::INFINITY);
        assert!(log(-1.0).is_nan() && log(f64::NAN).is_nan());
        assert_eq!(exp(710.0), f64::INFINITY);
        assert_eq!(exp(-746.0), 0.0);
        assert!(exp(f64::NAN).is_nan());
    }

    #[test]
    fn pow_is_close_to_std() {
        let mut rng = StreamRng::seed_from_u64(1);
        for _ in 0..20_000 {
            let base = rng.unit_f64() * 4.0;
            let exponent = (rng.unit_f64() - 0.5) * 60.0;
            let (ours, std) = (pow(base, exponent), base.powf(exponent));
            if std.is_finite() && std > 1e-300 {
                assert!(
                    ((ours - std) / std).abs() < 1e-13,
                    "pow({base}, {exponent}) = {ours}, std {std}"
                );
            }
        }
        assert_eq!(pow(0.0, 2.0), 0.0);
        assert_eq!(pow(0.0, 0.0), 1.0);
        assert_eq!(pow(0.5, 1.0), 0.5);
        assert_eq!(pow(3.0, 0.0), 1.0);
    }

    /// Fixed values: these must never change for the same major version, on any platform.
    #[test]
    fn portable_values() {
        let values = [
            exp(1.0),
            exp(-3.5),
            exp(0.1),
            pow(0.3, 1.0 / 21.0),
            pow(2.5, -20.0),
        ];
        assert_eq!(
            values.map(f64::to_bits),
            [
                4613303445314885482, // 2.7182818284590455, 1 ulp above e
                4584361024591036262, // 0.0301973834223185
                4607656066507473108, // 1.1051709180756477
                4606680541981188862, // 0.944280480021092
                4487727769212568094, // 1.0995116277759991e-8
            ],
            "the portable math changed, which breaks reproducibility"
        );
    }
}
