# Feature Gaps and Poisoning Attacks

## The Gap Between Features

A machine learning model takes an input $X$ and output $Y$. Essentially, the feature vector $X$ is transformed into another feature vector $Y$ after one or more linear (and non-linear) layers. Let's say, $Y=f(X)$. Here, $Y$ could be the output of the last layer such as the last fully-connected layer; it could also be that of an layer that is close to the last layer. 

Inituitively, humans percept and make decisions based on $X$. Hypothetically, machines make percept and make decisions based on $Y$. It seems to me at this moment that there is no way to prove this hypothesis. But the well-known task-related, explainable features that are generated by later layers of a convolutional neural network might support this hypothesis.  

*But is that a problem?* 

*Well, kind of.*

Let's say you have two images of a monkey and a lion, respectively. Say, you have $X_{monkey}$ and $X_{lion}$. You may have another image of monkey, say $X_{monkey}'$. In the input space, $X_{monkey}'$ is very similar to $X_{monkey}$ (and this is why your eye balls tell you this is a monkey). In other words, we say $||X_{monkey} - X_{monkey}'||_2^2 < ||X_{lion} - X_{monkey}'||_2^2$. 


Nevertheless, the machine's eyeballs may think $Y_{monkey}'$ is similar to $Y_{lion}$ instead of $Y_{monkey}$. In other words, we say $||Y_{monkey} - Y_{monkey}'||_2^2 > ||Y_{lion} - Y_{monkey}'||_2^2$. Or perhaps you can say $||f(X_{monkey}) - f(X_{monkey}')||_2^2 > ||f(X_{lion}) - f(X_{monkey}')||_2^2$. 

*But is that a problem?*

*Well, yes since you may generate $X_{monkey}'$ when you know $X_{monkey}$, $X_{lion}$, and $f()$.*

*But is that a problem?*

## Poisoning Attacks





## Backdorr Attacks

