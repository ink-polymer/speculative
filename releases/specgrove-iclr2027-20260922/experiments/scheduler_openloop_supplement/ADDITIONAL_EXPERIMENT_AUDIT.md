# Additional experiment audit

The two reviewer-critical gaps are now covered: matched TETRIS/ECHO-style
allocation controls and real wall-clock Poisson serving. The final serving study
uses 256 requests per trace, three seeds, and matched arrivals for all methods.

The present evidence is sufficient for the manuscript's scoped claims. If more
GPU time becomes available, the next useful additions are, in order:

1. a near-saturation point between 2 and 4 requests/s, because 2 requests/s is
   stable while 4 requests/s is overloaded;
2. a second hardware platform with a separately measured cost curve;
3. a longer-duration production workload with nonstationary arrivals;
4. planning-overhead scaling beyond 32 active requests.

Running the native TETRIS or ECHO stack is a different comparison: their original
proposal models, serving runtimes, cache policies, and batching paths do not match
the DFlash/DDTree backend used here. Such a result should be reported as an
end-to-end systems comparison, not substituted for the matched scheduler control.
