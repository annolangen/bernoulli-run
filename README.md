# Bernoulli Streaks

A Bernoulli trial is a random experiment with only two possible outcomes: success or failure. A Bernoulli streak is a sequence of successes, whose length $N$ has probability distribution:

$$
S(N) = p^N * (1-p)
$$

where $p$ is the probability of success. The cummulative distribution function (CDF) is:

$$
F(M) = \sum_{i=0}^{M}{S(N)} = 1 - p^{M+1}
$$

Its inverse is:

$$
F^{-1}(x) = \lceil \log_p(1-x)  \rceil - 1
$$

The inverse of the CDF is useful for generating random numbers with the underlying distribution.

<!--

The generating function is

$$
G(S)(x) = \sum_{N=0}^{\inf}{S(N)*x^N} = \frac{1-p}{1-p x}
$$

The expected length of a streak is:

$$
E[S(N)] = G(S)'(1) = \frac{(1-p) *p}{(1-p)^2} = \frac{p}{1-p}
$$

-->

Below are two algorithms that leverage Bernoulli streaks: Algorithm L for Reservoir Sampling and the Veach's consistent hashing.

## Reservoir Sampling

How should we uniformly sample $k$ items from a stream of unknown length $n$?
One way generates a random number for each item to decide whether to replace one of the $k$ previously sampled items.
For uniform sampling, after accepting the first $k$ items, the probability of rejecting the $i$-th item should be $p(i) = 1 - k/i$.

Rather than generating a random number for each item, we can generate a random number for each streak of rejections.
Think of $p(i) = 1 - k/i$ as the success probability of a Bernoulli trial.
Then the probility distribution of the length of the streak of successes starting at $i$ is:

$$
S(N,i) = p(i) * p(i+1) *...* p(i+N-1) * (1-p(i+N)) =
$$

$$
\frac{i - k}{i} * \frac{i + 1 - k}{i + 1} *...* \frac{i + N - 1 - k}{i + N - 1} * (1 - \frac{i + N - k}{i + N} ) =
$$

$$
\frac{i - k}{i} * \frac{i + 1 - k}{i + 1} *...* \frac{i + N - 1 - k}{i + N - 1} * \frac{k}{i + N} =
$$

$$
\frac{k*(i-1)!*(i+N-1-k)!}{(i + N)! * (i-k-1)!}
$$

I failed to find a closed form for the CDF of the length of the streak.
Related to [Algorithm L](https://en.wikipedia.org/wiki/Reservoir_sampling#Optimal:_Algorithm_L) for Reservoir Sampling, we might simply approximate the length of the streak starting at $i$ by using the inverse of the CDF of the Bernoulli streak distribution based on $p(i)$.

```cpp
for (int i = 0; i < k; i++) {
  reservoir[i] = stream[i];
}
for (int i = k; i < n;) {
  double r = rand();
  int streak = ceil(log(1 - r) / log(k / i)) - 1;
  i += streak;
  reservoir[i % k] = stream[i];
}
```

## Consistent Hashing

Sharding is one of the fundamental scaling techniques. Every key is mapped to a unique shard. A popular shard function is modulus, say shard(k, 10) = k % 10, for 10 shards. Consider what happens when we increase the number of shards to 11. Most keys would be remapped. Consistent hashing is a technique to minimizes the number of keys that need to be remapped when the number of shards changes. [Veach14](https://arxiv.org/pdf/1406.2294) proposed a consistent hashing algorithm that leverages Bernoulli streaks.

For a consistent hash, the shard function must may change only with a specific, low probability. For n > 1:

$$
p(shard(k, n) = shard(k, n-1)) = 1 - 1/n
$$

Using a psudo random number generator seeded with `k`, a slower algorithm might compute

```python
shard(k, n) = n-1 if n = 1 or random.next()*n > n-1 else shard(k, n-1)
```

Instead the algorithm is more like:

```python
shard(k, n) = n-1 if n = 1 or (r := random.next()) * n > n-1 else shard(k, n-jump(n, r))
```

where `jump(n, r)` is based on the inverse of the CDF of the Bernoulli streak distribution.

<!--

$$
S(N,i) \approx p(i)^N (1-p(i)) = \frac{(i-k)^N}{i^{N+1}} * k
$$

$$
\frac{i - k}{i} * \frac{i+1 - k}{i+1} *...* \frac{i+N-1 - k}{i+N-1} * (1-\frac{i+N - k}{i+N}) =
$$

$$
\frac{(i+N-1-k)!}{(i-k-1)!} * \frac{(i-1)!}{(i+N-1)!} * \frac{k}{i+N} =
$$

$$
\frac{(i+N-1-k)!}{(i-k-1)!} * \frac{(i-1)!}{(i+N)!} * k =
$$

The Reservoir Sampling algorithm P is a simple and elegant solution. It is a Bernoulli streak with a non-constant probability of success. The probability of success is $k/n$ for the first $k$ elements, and
$k/i$ for the $i$-th element, $i > k$. The expected length of the streak is:
 -->

