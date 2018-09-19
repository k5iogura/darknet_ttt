### Folding BatchNormalization ###

YOLO uses forward_convolutional_layer with BatchNormalization method.  
Original information is [here](http://machinethink.net/blog/object-detection-with-yolo/).  
Points are, 

```
out[j] = x[i]*w[0] + x[i+1]*w[1] + x[i+2]*w[2] + ... + x[i+k]*w[k] + b

        gamma * (out[j] - mean)
bn[j] = ---------------------- + beta
            sqrt(variance)
            
bn:= BatchNormalized value
```

YOLO has gamma, mean and variance a layer, "folding method" becames bellow,


```
               gamma[l]
factor[l] := -----------------
            sqrt(variance[l])
l:= layers
  
bn[j] := { factor[l] * out[j] } + { factor[l] * -mean[l] + beta[l] }

factor[l] * out[j] = ΣΣ(factor[l] * w[k] * x[i+k])

Here, w'[k]     = factor[l] * w[k]
      beta'[l] += factor[l] * -means[l]

Then,
bn[j]  := ΣΣ(w'[k] * x[i+k]) + beta'[j]

```
At stating of calcuration, darknet reads weigts, gamma, variance and beta from saved files, and
- [x] Darknet normally initialize out[j] to 0.0,  
- [x] At last of convolution, beta[l] is added to result of sGEMM.

"folding method" modifies normal darknet flow on bellow points,
- [x] "folding method" initialize out[j] to beta'[l],  
- [x] weights are initialized to w'[k].  

The gain of "folding" method is ``reducing number of sqrt and divsion floating point operations``.  
