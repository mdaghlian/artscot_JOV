import numpy as np
import sys

def cortical_permutation_test(
    a,b, subsample_fraction,
    n_subsamples= 10000,
    alternative= 'two-sided',
    random_seed = None
):
    """
    Permutation test for paired cortical surface data with spatial subsampling.

    Given array a (size N vx) and b (size N vx) is there a pairwise difference between them?

    Cortical surface data is upsampled from voxels to vertices
    To (approximately) correct for this we do the following:


    [1] diff = a - b

    [2] For a loop of size n_subsamples:

        [i] Randomly subsample vertices based on "subsample_fraction" 
            If n voxels project to m vertices, subsample_fraction = n / m                        
            we randomly sample "n" of the vertices; to approximately break the spatial dependencies
            diff_sub 


        [iii] Randomly flip the sign of diff_sub (diff_sub_flipped)

        [iv] Record with a boolean (TRUE, FALSE) whether:
            - "greater" : mean(diff_sub_flipped) > mean(diff_sub)
            - "less"    : mean(diff_sub_flipped) < mean(diff_sub)
            - "two-side": abs(mean(diff_sub_flipped)) > abs(mean(diff_sub))

    The "p-value" represents the proportion of subsamples meeting the criteria
    
    Parameters
    ----------
    a : np.ndarray, shape (n_vertices,)
        Parameter values for condition A
    b : np.ndarray, shape (n_vertices,)
        Parameter values for condition B (same vertices as A)
    subsample_fraction : float
        Fraction of vertices to randomly sample in each iteration.
        Should be small enough that sampled vertices are (approximately) spatially decorrelated.
        Typical values: 0.01 (1%) to 0.05 (5%)
    n_subsamples : int, default=10000
        Number of random subsamples to test.
        More subsamples = more stable p-value estimate.
    alternative : {'two-sided', 'greater', 'less'}, default='two-sided'
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    results : dict
        - 'pvalue': proportion of subsamples where null was as extreme as observed
        - 'mean_difference': average difference (A - B) across subsamples
        - 'std_difference': standard deviation of differences across subsamples
        - 'subsample_means': array of mean differences for each subsample
    
    Follows Aqil et al., 2021
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Validate inputs
    if len(a) != len(b):
        raise ValueError("a and b must have same length")
    
    n_vertices = len(a)
    n_subsample = max(1, int(n_vertices * subsample_fraction))
    
    
    # Precompute differences
    diff = a - b
    
    # Run subsampling permutation test
    subsample_means = np.zeros(n_subsamples)
    null_more_extreme = np.zeros(n_subsamples, dtype=bool)
    
    for i in range(n_subsamples):
        # Randomly subsample vertices
        subsample_idx = np.random.randint(0, n_vertices, size=n_subsample)
        diff_sub = diff[subsample_idx]
        
        # Compute observed weighted mean difference
        observed_mean = np.mean(diff_sub)
        subsample_means[i] = observed_mean
        
        # Generate ONE null sample by randomly flipping signs
        # Under H0 (no difference), sign of each vertex's difference is arbitrary
        random_signs = np.sign(np.random.rand(n_subsample) - 0.5)
        null_diff = random_signs * diff_sub
        null_mean = np.mean(null_diff)
        
        # Check if null is as extreme as observed
        if alternative == 'two-sided':
            null_more_extreme[i] = np.abs(null_mean) >= np.abs(observed_mean)
        elif alternative == 'greater':
            null_more_extreme[i] = null_mean >= observed_mean
        elif alternative == 'less':
            null_more_extreme[i] = null_mean <= observed_mean
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    
    # Compute final p-value
    pvalue = np.mean(null_more_extreme)
    
    # Compute effect size (standardized mean difference)
    mean_diff = np.mean(subsample_means)
    std_diff = np.std(subsample_means)
    effect_size = mean_diff / std_diff if std_diff > 0 else 0
    
    return {
        'pvalue': pvalue,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'subsample_means': subsample_means,
        'effect_size': effect_size,
        'n_subsample_vertices': n_subsample,
        'n_subsamples': n_subsamples,
        'subsample_fraction': subsample_fraction
    }


def cortical_permutation_test_fast(
    a, b, subsample_fraction,
    n_subsamples=10000,
    alternative='two-sided',
    random_seed=None
):
    """
    Bit-identical, vectorised version of cortical_permutation_test.

    numpy's legacy global RandomState is a single mutable stream: each loop
    iteration in cortical_permutation_test draws from it twice, in a fixed
    order (randint then rand), and the two draw types interleave into that
    stream in a way that depends on call order. So matching its output
    exactly means replaying that exact sequence of randint/rand calls one
    subsample at a time -- that part cannot be batched. Everything
    downstream of the draws (indexing, means, sign multiplication,
    thresholding) does NOT touch the random state, so it's pulled out of the
    loop and vectorised over all n_subsamples at once. That's where the
    speedup comes from: the loop body shrinks from ~6 numpy calls to 2.

    Uses a local np.random.RandomState instead of reseeding the global
    np.random state, so it's safe to call from multiple threads at once
    (e.g. joblib Parallel(prefer='threads')) without one call's draws
    corrupting another's. A local RandomState(seed) draws the exact same
    sequence as seeding the global state and calling np.random.randint /
    np.random.rand, so this is still bit-identical to cortical_permutation_test.

    Parameters / Returns: identical to cortical_permutation_test.
    """
    if len(a) != len(b):
        raise ValueError("a and b must have same length")
    if alternative not in ('two-sided', 'greater', 'less'):
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    a = np.asarray(a)
    b = np.asarray(b)
    n_vertices = len(a)
    n_subsample = max(1, int(n_vertices * subsample_fraction))

    # Precompute differences
    diff = a - b

    random_state = np.random.RandomState(random_seed)

    # Replay the exact same sequence of draws as the original loop (this is
    # the only part that must stay a loop, to preserve RNG state order)
    all_idx = np.empty((n_subsamples, n_subsample), dtype=np.intp)
    all_rand = np.empty((n_subsamples, n_subsample), dtype=np.float64)
    for i in range(n_subsamples):
        all_idx[i] = random_state.randint(0, n_vertices, size=n_subsample)
        all_rand[i] = random_state.rand(n_subsample)

    # Everything below is pure downstream math -- vectorised over all subsamples
    diff_sub = diff[all_idx]                          # (n_subsamples, n_subsample)
    subsample_means = diff_sub.mean(axis=1)

    random_signs = np.sign(all_rand - 0.5)
    null_means = (random_signs * diff_sub).mean(axis=1)

    if alternative == 'two-sided':
        null_more_extreme = np.abs(null_means) >= np.abs(subsample_means)
    elif alternative == 'greater':
        null_more_extreme = null_means >= subsample_means
    else:
        null_more_extreme = null_means <= subsample_means

    # Compute final p-value
    pvalue = np.mean(null_more_extreme)

    # Compute effect size (standardized mean difference)
    mean_diff = np.mean(subsample_means)
    std_diff = np.std(subsample_means)
    effect_size = mean_diff / std_diff if std_diff > 0 else 0

    return {
        'pvalue': pvalue,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'subsample_means': subsample_means,
        'effect_size': effect_size,
        'n_subsample_vertices': n_subsample,
        'n_subsamples': n_subsamples,
        'subsample_fraction': subsample_fraction
    }