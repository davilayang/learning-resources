# Statistic in Action
<!-- markdownlint-disable MD034 -->

## Communication

- Start with low-stakes meetings
- Repeat back what someone said
- Lay out a set of options and discuss with stakeholders

## A/B Testing

- Sample
- Inferential Statistics
- Decide the Sample Size
- Construct Confidence Interval
- Determine the Statistical Significance
- Average height of the entire population of giraffes is a "parameter"
  - Average height of a random sample of 100 giraffes is a "statistic"
  - Use the known value of a sample statistic to estimate the unknown value of a population parameter
  - This is called "Inferential Statistics"
    - I.e. using sample mean to inference population mean

Knowing your dataset

1. Know the "center" of the dataset
   - Measure of Central Tendency
   - Mean, median, mode
     - Mean is easily affected by outliers, use median is there're outliers
     - Mode is useful with categorical data
2. Know "how spread out the other values" of the dataset
   - Measure of Dispersion
   - Range eVariation and Standard Deviation
     - Range as (Max - Min)
     - Standard Deviation measures **value spreads from the mean**
       - For every data point, calculates distance from the mean
     - Variance measures average of squared difference of each data point from the mean 
     - TODO: know the formulas as basic!
2. Know "what kind of statistical distribution" fits the data
   - Normal? Poisson? ...etc
3. Decide which part to explore further from above information
4. Measure of position
   - Percentiles: the value below which a percentage of data falls
     - e.g. 67% is at 0.7, meaning 67% of the data points is below "0.7"
   - Quartiles: dividing dataset into 4 equal parts
     - Q1, first quartile is the middle value it the first half of the dataset
     - Q2, also the median, or the 50th Percentile
     - Q3, the 75th Percentile
   - Inter-Quartile Range: Q3 - Q1
     - I.e. measures the range of the middle 50% of the dataset
   - Five number summary: min, Q1, median, Q3, max

## Probabilities

- Objective vs Subjective
  - Former is based on statistics, experiments and mathematical measurements
    - Classical Probability, based on formal reasoning
      - e.g. flipping a coin
    - Empirical Probability, based on experimental or historical data
      - e.g. predicting weather tomorrow
      - used in Inferential Statistics  
  - Later is based on personal feeling or judgement
- Mutually Exclusive Events
  - **Two events are exclusive if they cannot happen at the same time**
    - E.g. not rain vs rain
  - Additional Rule
    - For two mutually exclusive events, P(A or B) = P(A) + P(B)
- Independent Events
  - Two event are independent if the occurrence of one does not change the probability of the other event
  - I.e. one event does not affect the outcome of the other events
  - Multiplication Rule
    - For two independent events, P(A and B) = P(A) * P(B)
- Conditional Probability (for Dependent events)
  - Dependent Events
    - First event affects the outcome of the second event
  - For two dependent events, P(A and B) = P(A) * P(B | A)
    - P(B | A) = P(A and B) / P(A)
- TODO: Rephrase the business questions as a conditional probability question?
  - Given a user has enrolled, what is the probability that user will get started on a course?
  - Given an user has completed N lessons, probability of that user will be retained in the following month?
  - ...?
- Bayes Theorem
  - P(A | B) = (P(B | A) * P(A)) / P(B)
    - Prior Probability, Probability before new data is collected
    - Posterior Probability, Updated Probability with the new data
  - Expanded https://www.coursera.org/learn/the-power-of-statistics/lecture/eFn2p/the-expanded-version-of-bayess-theorem

Discrete Probability Distributions

- Binomial Distribution
  - Represents experiments with repeated trials and each two possible outcomes
  - TODO: Rephrase business questions with only two outcomes as binomial distribution?
  - Given the exact probability of an event happening
    - To find the probability of the event happening a certain number of times in a repeated trial
- Poisson Distribution
  - TODO: Rephrase business questions with only two outcomes as binomial distribution?
  - Represents a certain number of events will occur during a specific time period
    - Or during a specific space (distance, area, volume ...etc)
  - Pre-requirements
    - Countable number of events (discrete)
    - Known value of "mean number of events that occurred during a specific timer period"
    - Each event is independent
  - Examples
    - For a delivery service, number of orders received in a given 30 minutes periods
  - Given the average probability of an event happening for a specific time period
    - To find the probability of a certain number of events happening in the time period
- https://www.coursera.org/learn/the-power-of-statistics/supplement/4RNyV/discrete-probability-distributions


Continuous Probability Distribution

- Normal Distribution
- Z-Score, or "Standard Score"
  - Measure of how many standard deviations below or above the population mean a data point is
  - ==0, means the value equals to the population mean
  - Z = (x - population mean) / (population standard deviation)
  - Help in standardizing the data
  - Based on Standard Normal Distribution, i.e. normal distribution with mean=0 and std=1
  - NOTE: standardization to allow comparing values from different data distributions
    - E.g. with the Weekly Briefing content position problem,
      - it's actually about "standardizing" the click rates of each position

## Sampling

- Probabilistic Sampling
  - Simple random sampling
  - Stratified random sampling (strata)
  - Cluster random sampling (the cluster)
    - divide a population into clusters
    - randomly select certain clusters
    - include all members from the chosen clusters in the sample
  - Systematic random sampling (ordered and interval)
    - put every member of a population into an ordered sequence
    - choose a random starting point in the sequence
    - select members for your sample at regular intervals
- Non-probabilistic Sampling
  - Convenience sampling (only by convenience to the researcher)
  - Voluntary response sampling
    - Only by respondents who are willingly to respond
  - Snowball sampling
    - Only by respondents and their close personnels
  - Purposive sampling
    - Only by certain research purpose
- Sampling Distributions
  - Standard Error
    - The spread of sampling distribution mean, or the variability
    - Larger meaning the sample means are more spread out (not closer to the true population mean)
    - standard error = sample standard deviation / square root of sample size
  - Central Limit Theorem
    - Sampling distribution of the mean
      - approaches a normal distribution as the sample size increases
      - i.e. assumes the shape of bell-curve as the sample size increases
      - with enough larger sample size, the sample mean will roundly equal to population mean
    - Population distribution may not be the same as normal distribution
      - but the sample mean will approach bell-shape
    - So even with skewed population distribution, the sampling distribution will still approach normal distribution. Why? Sampling distribution is the distribution of sample means (of multiple samples).
      - And the mean of the sampling distribution will give good estimate of the population mean
    - https://www.coursera.org/learn/the-power-of-statistics/supplement/YVpN8/infer-population-parameters-with-the-central-limit-theorem
    - https://www.coursera.org/learn/the-power-of-statistics/supplement/T0Z3d/the-sampling-distribution-of-the-mean
- Population Proportion
  - Percentage of individuals (or elements) in a population 
    - that shares a certain characteristic

## Confidence Intervals

- Expression of Uncertainty, instead of point estimate
- Confidence Interval
  - Margin Error = Z-score * Standard Error
    - Z-score: the distance of a data point from the population mean in a standard normal distribution
- Confidence Level
  - Likelihood a particular sampling method will produce a confidence interval that include the population parameter
  - Usually at 95%
- Interpretation 
  - 95% of intervals will capture the population mean
    - i.e. 5% are not captured
    - expect 19 out of 20 of the intervals will capture the population mean
  - Miss-interpretation
    - 95% of the intervals will capture the "sample" mean
    - 95% of the data points will fall within the interval
  - https://www.coursera.org/learn/the-power-of-statistics/lecture/BWAiF/interpret-confidence-intervals
  - https://www.coursera.org/learn/the-power-of-statistics/supplement/GxY7A/confidence-intervals-correct-and-incorrect-interpretations
- Steps
  1. Identify a sample statistic (e.g. mean, median, proportion...etc)
  2. Choose a confident level (usually 95%)
  3. Find the margin of error
     - Range of values about and below the sample statistic
     - Margin of Error = Z-Score * Standard Error
       - Z-Score, the distance of a data point from the population mean in a standard normal distribution
         - e.g. if Z-score = 1, then it's 1 standard deviation above the mean
       - Standard Error, measures the variability of the sample statistic
         - For Proportion: Square root of ( `sample proportion` * (1 - `sample proportion`)  / `sample size`)
         - For Mean: population std / Square root of (`sample size`)
  4. Calculate the Confidence Interval
     - Upper & Lower limit
- https://www.coursera.org/learn/the-power-of-statistics/lecture/VCasN/introduction-to-confidence-intervals
- https://www.coursera.org/learn/the-power-of-statistics/supplement/cdOx7/construct-a-confidence-interval-for-a-small-sample-size

## Hypothesis Testing

- https://www.coursera.org/learn/the-power-of-statistics/lecture/1Ripn/introduction-to-hypothesis-testing
- Steps
  1. State the null hypothesis and the alternative hypothesis.
     - Null: assumed to be True, observed happened only by chance and no effect from treatment
     - Alternative: observed data does not happen by chance, having effect from treatment
     - https://www.coursera.org/learn/the-power-of-statistics/supplement/p1cZU/differences-between-the-null-and-alternative-hypotheses
  2. Choose a significance level.
  3. Find the p-value.
  4. Reject or fail to reject the null hypothesis.
- Errors
  - **Type I Error**
    - False Positive
      - Falsely classified as positive, while true label is negative (classification problem)
      - Falsely considered alternative as true (positive), while it should false; and rejected null
    - Reject the null hypothesis when it’s actually true
    - significance level, or alpha (α), represents the probability of making a Type I error
    - NOTE: "I" and then "II", same as "Positive" then "Negative"
  - **Type II Error**
    - False Negative
      - Falsely classified as negative, while true label is positive (classification problem)
      - Falsely considered alternative as false (negative), while it should be true; and did not reject null
    - Fail to reject the null hypothesis when it’s actually false 
    - Pobability of making a Type II error is called beta (β), 
      - beta is related to the power of a hypothesis test (power = 1- β)
      - Power refers to the likelihood that a test can correctly detect a real effect when there is one
  - https://www.coursera.org/learn/the-power-of-statistics/supplement/Scyf1/type-i-and-type-ii-errors
- One Sample Test
  - A population parameter is equal to a specific value or not
    - E.g. average sales revenue (of samples) equals to a target value or not
    - E.g. stock portfolio average rate equals to benchmark
  - Steps.
    1. State the null hypothesis and the alternative hypothesis
       - Null: population means equals to the observed value
       - Alt: 
    2. Choose a significance level.
    3. Find the p-value.
    4. Reject or fail to reject the null hypothesis.
- Two Sample Test
  - Two population parameters are equal to each other or not
    - E.g. in A/B Testing, Group A vs Group B
  - Assumptions
    - Two samples are independent of each other
    - Samples are drawn randomly from a normally distributed population
    - Population standard deviation is unknown (thus using t-test)
      - Using z-test if this is known
  - TODO: add hypothesis testing to your data analyst arsenal
- P-Value
  - Probability of observing a difference in the results
    - as or more extreme than the difference observed when null is true
  - IF "P-Value < Significance Level", THEN "Reject Null Hypothesis"
  - E.g. p-value is the probability of observing a difference
    - that is two minutes or greater if the null hypothesis is true
  - TO-DO: How do you rephrase this?
  - Calculations
    - From "Test Statistic"
      - a value that shows how closely your observed data matches the distribution expected under the null hypothesis
      - assume null is true, and known mean, then the observed data follows normal distribution
      - shows where the observed data falls on the distribution
      - TODO: what is test statistic?
    - Z-score
      - Measure of how many standard deviations below or above the population mean
      - Tell where the value lies on a normal distribution
      - Z-score = (Sample Mean - Population Mean) / (population std / square root of sample size)
    - https://www.coursera.org/learn/the-power-of-statistics/lecture/Kv9dl/one-sample-test-for-means
