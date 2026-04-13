# Actuarial Statistics Project  
## Confidence Intervals in Health Insurance Claims

This project looks at a simple question that actually matters a lot in actuarial work:

**Do standard confidence intervals actually achieve their intended coverage when applied to realistic medical claim distributions?**

---

## Motivation

I’m pursuing actuarial science, and on the health side, the goal is to estimate expected costs so insurance companies can price correctly.

- If we **underestimate** → the company loses money  
- If we **overestimate** → premiums are too high and you lose customers  

Confidence intervals are supposed to measure uncertainty, but most of what we learn assumes normality or “large enough” sample sizes. In healthcare data, that usually isn’t true.

---

## Background

In health insurance, costs come in the form of **claims**:

- Most are small (doctor visits, prescriptions)  
- A few are extremely large (surgeries, cancer treatment, hospital stays)  

This leads to:
- Right-skewed distributions  
- Heavy tails  
- Catastrophic outliers  

Because of this, actuaries use distributions like:
- Gamma  
- Lognormal  
- Mixtures of distributions  

---

## Project Approach

Instead of using real data (where the true mean is unknown), I built a **Monte Carlo simulation**.

This allows me to:
- Set the true mean to 1  
- Repeatedly sample data  
- Measure how often confidence intervals actually contain the true mean  

### Coverage

For a 95% confidence interval, we expect about 95% of intervals to contain the true mean.

If it’s lower than that, we are **underestimating risk**.

---

## Distributions Used

- **Normal (baseline)** – verifies methods work under ideal assumptions  
- **Gamma** – moderate skew (realistic for some claims)  
- **Lognormal (moderate)** – common in insurance  
- **Lognormal (high skew)** – models high-risk populations  
- **Mixture (most realistic)** – 95% small claims, 5% catastrophic  

---

## Methods

For each distribution and sample size:

- 10,000 simulations  
- Confidence intervals:
  - Z-interval  
  - T-interval  
  - Bootstrap (499 resamples)  

Metrics evaluated:
- Coverage  
- Bias  
- Interval width  
- Mean Squared Error (MSE)  
- Type 1 error  

---

## Sample Sizes

- **n = 25, 50** → small groups (high variability)  
- **n = 100, 250, 500** → larger groups (asymptotic behavior)

---

## Key Results

### Coverage Results

![Coverage Plot](simulation_plots\plot2_coverage.png)

- Normal behaves as expected (~95%)  
- Moderate skew improves with sample size  
- High skew never fully reaches 95%  
- Mixture distribution breaks completely at larger n  

---

### Confidence Interval Width

![CI Width Plot](simulation_plots\plot3_width.png)

- Wider intervals at small sample sizes  
- Mixture appears “accurate” early only because intervals are very wide  
- As intervals shrink, coverage drops  

---

### Main Takeaways

- Confidence intervals work under normal assumptions  
- Moderate skew requires larger samples  
- Heavy skew leads to consistent undercoverage  
- Mixture distributions introduce bias that does not go away  

---

## Actuarial Implications

If a 95% confidence interval only captures:

- 90% → underestimating risk  
- 80% → serious pricing issues  

This can lead to:
- Insufficient reserves  
- Mispricing insurance products  
- Financial and regulatory risk  

**Conclusion:**

Standard confidence intervals are not always reliable for healthcare claims, especially with skewed or mixture distributions.

---

## Technical Insights

- The Central Limit Theorem does not guarantee good performance in realistic settings  
- Skewness and tail behavior directly affect inference  
- Mixture distributions create structural bias  
- Bootstrap methods do not automatically fix these issues  

---

## Tools Used

- Python  
- Monte Carlo simulation  
- Statistical modeling  
- Bootstrap methods  

---

## Final Thoughts

This project is more of a stress test of statistical methods than a modeling exercise.

It shows that:
- Methods that work in theory don’t always work in practice  
- In actuarial work, understanding the data matters just as much as applying formulas  

---