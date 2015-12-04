# machl
Vectorized machine learning algorithms.

### Objective Representation
1. Hypothesis Testing: model the trendline function.

2. Parameter Fitting: weigh the correlation features.

3. Algorithm Training: minimize the cost function.

4. Decision Making & Problem Solving: classify, recommend, predict.

### Convergence Function Manual
* The difference between each hypothesized and actual value, aggregated, is the total error. This represents the model's cost.
 
* Minimize the cost without overfitting the data by determining best parameters via algorithmic representation and iteration, and by not applying too many features.

* Use a fixed learning rate, since minimum-approaching derivatives decrease by smaller increments. Note that too small an alpha means a slower algo, and too large means lower precision and possibility of convergence failure.

* Too little features = probable underfit. Multivariate implementation scales features to fit between -1 and 1 via mean normalization, where the mean ~= zero.

### Debugging+Optimization: Considerations Where Applicable
* Increase training examples.
* Revise number of feature sets and polynomial complexity.
* Initialize feature weights randomly to break training symmetry.
* Experiment with different values for regularization parameter lambda. The introduction of additional quadratic features for linear separability (e.g. decision boundaries) requires an optimal complexity penalty.
* If features not many, consider analytic over algorithmic approach. A matrix requires no alpha or iterations.
* Training set learning curve convergence errors:
 * Wide gap = high variance.
 * Narrow gap = high bias.
* Error analysis:
 * Precision vs Recall (True/False Positives+Negatives):
   * f-score = 2(PR/P+R).
 * Training / Validation / Testing Sets.

### Supervised v Unsupervised Learning
While the former solves for explicit unknowns, via discreet or continuous fashion, unsupervised techniques identify patterns autonomously. Possible algorithms differ variably from the aforementioned, computing metrics like similarity (e.g. via gaussian kernel), and distance (e.g. via k-means), for detect, rank, cluster, and extract functions.