Download Link: https://assignmentchef.com/product/solved-ift6135-assigment-2
<br>
Question 1 (4-4-4). In this question you will demonstrate that an estimate of the first moment of the gradient using an (exponential) running average is equivalent to using momentum, and is biased by a scaling factor. The goal of this question is for you to consider the relationship between different optimization schemes, and to practice noting and quantifying the effect (particularly in terms of bias/variance) of <em>estimating </em>a quantity.

Let <em>g</em><em><sub>t </sub></em>be an unbiased sample of gradient at time step <em>t </em>and ∆<em>θ</em><em><sub>t </sub></em>be the update to be made. Initialize <em>v</em><sub>0 </sub>to be a vector of zeros.

<ol>

 <li>For <em>t </em>≥ 1, consider the following update rules:

  <ul>

   <li>SGD with momentum:</li>

  </ul></li>

</ol>

<em>v</em>         ∆<em>θ</em><em><sub>t </sub></em>= −<em>v</em><em><sub>t</sub></em>

where and <em>α </em>∈ (0<em>,</em>1).

<ul>

 <li>SGD with running average of <em>g</em><em><sub>t</sub></em>:</li>

</ul>

<em>v</em><em><sub>t </sub></em>= <em>β</em><em>v</em><em><sub>t</sub></em><sub>−1 </sub>+ (1 − <em>β</em>)<em>g</em><em><sub>t                          </sub></em>∆<em>θ</em><em><sub>t </sub></em>= −<em>δ</em><em>v</em><em><sub>t</sub></em>

where <em>β </em>∈ (0<em>,</em>1) and <em>δ &gt; </em>0.

Express the two update rules recursively (∆<em>θ</em><em><sub>t </sub></em>as a function of ∆<em>θ</em><em><sub>t</sub></em><sub>−1</sub>). Show that these two update rules are equivalent; i.e. express  as a function of (<em>β,δ</em>).

<ol start="2">

 <li>Unroll the running average update rule, i.e. express <em>v</em><em><sub>t </sub></em>as a linear combination of <em>g</em><em><sub>i</sub></em>’s (1 ≤ <em>i </em>≤ <em>t</em>).</li>

 <li>Assume <em>g</em><em><sub>t </sub></em>has a stationary distribution independent of <em>t</em>. Show that the running average is biased, i.e. E[<em>v</em><em><sub>t</sub></em>] 6= E[<em>g</em><em><sub>t</sub></em>]. Propose a way to eliminate such a bias by rescaling <em>v</em><em><sub>t</sub></em>.Question 2 (7-5-5-3). The point of this question is to understand and compare the effects of different regularizers (specifically dropout and weight decay) on the weights of a network. Consider a linear regression problem with input data <em>X </em>∈ R<em><sup>n</sup></em><sup>×<em>d</em></sup>, weights <em>w </em>∈ R<em><sup>d</sup></em><sup>×1 </sup>and targets <em>y </em>∈ R<em><sup>n</sup></em><sup>×1</sup>. Suppose that dropout is applied to the input (with probability 1−<em>p </em>of dropping the unit i.e. setting it to 0). Let <em>R </em>∈ R<em><sup>n</sup></em><sup>×<em>d </em></sup>be the dropout mask such that <em>R</em><em><sub>ij </sub></em>∼ Bern(<em>p</em>) is sampled i.i.d. from theBernoulli distribution.For a squared error loss function with dropout, we then have:

  <ol>

   <li>Let Γ be a diagonal matrix with . Show that the <em>expectation (over R) </em>of the loss function can be rewritten as E[<em>L</em>(<em>w</em>)] = ||<em>y </em>− <em>p</em><em>Xw</em>||<sup>2 </sup>+ <em>p</em>(1 − <em>p</em>)||Γ<em>w</em>||<sup>2</sup>. <em>Hint: Note we are trying to find the expectation over a squared term and use </em>Var(<em>Z</em>) = E[<em>Z</em><sup>2</sup>] − E[<em>Z</em>]<sup>2</sup><em>.</em></li>

   <li>Show that the solution <em>w</em><sup>dropout </sup>that minimizes the expected loss from question 2.1 satisfies</li>

  </ol><em>pw</em>dropout = (<em>X</em>&gt;<em>X </em>+ <em>λ</em>dropoutΓ2)−1<em>X</em>&gt;<em>y</em>where <em>λ</em><sup>dropout </sup>is a regularization coefficient depending on <em>p</em>. How does the value of <em>p </em>affect the regularization coefficient, <em>λ</em><sup>dropout </sup>?

  <ol start="3">

   <li>Express the loss function for a linear regression problem without dropout and with <em>L</em><sup>2 </sup>regularization, with regularization coefficient <em>λ<sup>L</sup></em><sup>2</sup>. Derive its closed form solution <em>w</em><em><sup>L</sup></em><sup>2</sup>.</li>

   <li>Compare the results of 2.2 and 2.3: identify specific differences in the equations you arrive at, and discuss qualitatively what the equations tell you about the similarities and differences in the effects of weight decay and dropout (1-3 sentences).Question 3 (6-10-2). The goal of this question is for you to understand the reasoning behind different parameter initializations for deep networks, particularly to think about the ways that the initialization affects the activations (and therefore the gradients) of the network. Consider the following equation for the <em>t</em>-th layer of a deep network:<em>h</em>(<em>t</em>) = <em>g</em>(<em>a</em>(<em>t</em>))                     <em>a</em>(<em>t</em>) = <em>W </em>(<em>t</em>)<em>h</em>(<em>t</em>−1) + <em>b</em>(<em>t</em>)where <em>a</em><sup>(<em>t</em>) </sup>are the pre-activations and <em>h</em><sup>(<em>t</em>) </sup>are the activations for layer <em>t</em>, <em>g </em>is an activation function, <em>W </em><sup>(<em>t</em>) </sup>is a <em>d</em><sup>(<em>t</em>) </sup>× <em>d</em><sup>(<em>t</em>−1) </sup>matrix, and <em>b</em><sup>(<em>t</em>) </sup>is a <em>d</em><sup>(<em>t</em>) </sup>× 1 bias vector. The bias is initialized as a constant vector <em>b</em><sup>(<em>t</em>) </sup>= [<em>c,..,c</em>]<sup>&gt; </sup>for some <em>c </em>∈ R, and the entries of the weight matrix are initialized by samplingi.i.d. from a Gaussian distribution.Your task is to design an initialization scheme that would achieve a vector of pre-activations at layer <em>t </em>whose elements are zero-mean and unit variance (i.e.:  and ,1 ≤ <em>i </em>≤ <em>d</em><sup>(<em>t</em>)</sup>) for the assumptions about either the activations or pre-activations of layer <em>t</em>−1 listed below. Note we are not asking for a general formula; you just need to provide one setting that meets these criteria (there are many possiblities).

    <ol>

     <li>First assume that the activations of the previous layer satisfy and for 1 ≤ <em>i </em>≤ <em>d</em><sup>(<em>t</em>−1)</sup>. Also, assume entries of <em>h</em><sup>(<em>t</em>−1) </sup>are uncorrelated (the answer should not depend on <em>g</em>).

      <ul>

       <li>Show Var(<em>XY </em>) = Var(<em>X</em>)Var(<em>Y </em>) + Var(<em>X</em>)E[<em>Y </em>]<sup>2 </sup>+ Var(<em>Y </em>)E[<em>X</em>]<sup>2 </sup>when <em>X </em>⊥ <em>Y</em></li>

       <li>Write and in terms of.</li>

       <li>Give values for <em>c</em>, <em>µ</em>, and <em>σ</em><sup>2 </sup>as a function of <em>d</em><sup>(<em>t</em>−1) </sup>such that and for 1 ≤ <em>i </em>≤ <em>d</em><sup>(<em>t</em>)</sup>.</li>

      </ul></li>

     <li>Now assume that the pre-activations of the previous layer satisfy</li>

    </ol>1 and  has a symmetric distribution for 1 ≤ <em>i </em>≤ <em>d</em><sup>(<em>t</em>−1)</sup>. Assume entries of <em>a</em><sup>(<em>t</em>−1) </sup>are uncorrelated. Consider the case of ReLU activation: <em>g</em>(<em>x</em>) = max{0<em>,x</em>}.

    <ul>

     <li>Derive</li>

     <li>Using the result from (a), give values for <em>c</em>, <em>µ</em>, and <em>σ</em><sup>2 </sup>as a function of <em>d</em><sup>(<em>t</em>−1) </sup>such that and  for 1 ≤ <em>i </em>≤ <em>d</em><sup>(<em>t</em>)</sup>.</li>

     <li>What popular initialization scheme has this form?</li>

     <li>Why do you think this initialization would work well in practice? Answer in 1-2 sentences.</li>

    </ul>

    <ol start="3">

     <li>For both assumptions (1,2) give values such that and</li>

    </ol>.Question 4 (4-6-6). This question is about normalization techniques.

    <ol>

     <li>Batch normalization, layer normalization and instance normalization all involve calculating the mean <em>µ </em>and variance <em>σ</em><strong><sup>2 </sup></strong>with respect to different subsets of the tensor dimensions. Given the following 3D tensor, calculate the corresponding mean and variance tensors for each normalization technique: <em>µ</em><em>batch</em>, <em>µ</em><em>layer</em>, <em>µ</em><em>instance</em>, <em>σ</em><em>batch</em>2 , <em>σ</em><em>layer</em>2 , and <em>σ</em><em>instance</em>2 .</li>

    </ol>The size of this tensor is 4 x 2 x 3 which corresponds to the batch size, number of channels, and number of features respectively.

    <ol start="2">

     <li>For the next two subquestions, we consider the following parameterization of a weight vector <em>w</em>:</li>

    </ol><em>u</em><em>w </em>:= <em>γ</em>||<em>u</em>||where <em>γ </em>is scalar parameter controlling the magnitude and <em>u </em>is a vector controlling the direction of <em>w</em>.Consider one layer of a neural network, and omit the bias parameter. To carry out batch normalization, one normally standardizes the preactivation and performs elementwise scale and shift where <em>y </em>= <em>u</em><sup>&gt;</sup><em>x</em>. Assume the data <em>x </em>(a random vector) is whitened (Var(<em>x</em>) = <em>I</em>)and centered at 0 (E[<em>x</em>] = <strong>0</strong>). Show that <em>y</em>ˆ = <em>w</em><sup>&gt;</sup><em>x </em>+ <em>β</em>.

    <ol start="3">

     <li>Show that the gradient of a loss function <em>L</em>(<em>u</em><em>,γ,β</em>) with respect to <em>u </em>can be written in the form</li>

    </ol>∇<em><sub>u</sub></em><em>L </em>= <em>s</em><em>W </em><sup>⊥</sup>∇<em><sub>w</sub></em><em>L </em>for some <em>s</em>, where <em>W </em><sup>⊥ </sup>. Note that<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> <em>W </em><sup>⊥</sup><em>u </em>= <strong>0</strong>.<a href="#_ftnref1" name="_ftn1">[1]</a> . As a side note: <em>W</em><sup>⊥ </sup>is an orthogonal complement that projects the gradient away from the direction of <em>w</em>, which is usually (empirically) close to a dominant eigenvector of the covariance of the gradient. This helps to condition the landscape of the objective that we want to optimize.Question 5 (4-6-4). This question is about activation functions and vanishing/exploding gradients in recurrent neural networks (RNNs). Let <em>σ </em>: R → R be an activation function. When the argument is a vector, we apply <em>σ </em>element-wise. Consider the following recurrent unit:<em>h</em><em><sub>t </sub></em>= <em>W</em><em>σ</em>(<em>h</em><em><sub>t</sub></em><sub>−1</sub>) + <em>Ux</em><em><sub>t </sub></em>+ <em>b</em>

    <ol>

     <li>Show that applying the activation function in this way is equivalent to the conventional way of applying the activation function: <em>g</em><em><sub>t </sub></em>= <em>σ</em>(<em>Wg</em><em><sub>t</sub></em><sub>−1 </sub>+ <em>Ux</em><em><sub>t </sub></em>+ <em>b</em>) (i.e. express <em>g</em><em><sub>t </sub></em>in terms of <em>h</em><em><sub>t</sub></em>). More formally, you need to prove it using mathematical induction. You only need to prove the induction step in this question, assuming your expression holds for time step <em>t </em>− 1.</li>

    </ol><sup>∗</sup>2. Let ||<em>A</em>|| denote the <em>L</em><sub>2 </sub>operator norm<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> of matrix <em>A </em>(||<em>A</em>|| := max<em><sub>x</sub></em><sub>:||<em>x</em></sub><sub>||=1 </sub>||<em>Ax</em>||). Assume <em>σ</em>(<em>x</em>) has bounded derivative, i.e. |<em>σ</em><sup>0</sup>| ≤ <em>γ </em>for some <em>γ &gt; </em>0 and for all <em>x</em>. We denote as <em>λ</em><sub>1</sub>(·) the largest eigenvalue of a symmetric matrix. Show that if the largest eigenvalue of the weights is bounded by  for some 0 ≤ <em>δ &lt; </em>1, gradients of the hidden state will vanish over time, i.e.as <em>T </em>→ ∞Use the following properties of the <em>L</em><sub>2 </sub>operator norm||<em>AB</em>|| ≤ ||<em>A</em>||||<em>B</em>||            and             ||<em>A</em>|| = <sup>p</sup><em>λ</em><sub>1</sub>(<em>A</em><sup>&gt;</sup><em>A</em>)

    <ol start="3">

     <li>What do you think will happen to the gradients of the hidden state if the condition in the previous question is reversed, i.e. if the largest eigenvalue of the weights is larger than ? Is this condition <em>necessary </em>or <em>sufficient </em>for the gradient to explode? (Answer in 1-2 sentences).</li>

    </ol><a href="#_ftnref1" name="_ftn1">[1]</a> . The <em>L</em><sub>2 </sub>operator norm of a matrix <em>A </em>is is an <em>induced norm </em>corresponding to the <em>L</em><sub>2 </sub>norm of vectors. You can try to prove the given properties as an exercise.Question 6 (4-8-8). Consider the following Bidirectional RNN:<em>h</em><em>h</em>where the superscripts <em>f </em>and <em>b </em>correspond to the forward and backward RNNs respectively and <em>σ </em>denotes the logistic sigmoid function. Let <em>z</em><em><sub>t </sub></em>be the true target of the prediction <em>y</em><em><sub>t </sub></em>and consider the sum of squared loss <em>L </em>= <sup>P</sup><em><sub>t </sub></em><em>L<sub>t </sub></em>where.In this question our goal is to obtain an expression for the gradients ∇<em><sub>W</sub></em>(<em><sub>f</sub></em><sub>)</sub><em>L </em>and ∇<em><sub>U</sub></em>(<em><sub>b</sub></em><sub>)</sub><em>L</em>.

    <ol>

     <li>First, complete the following computational graph for this RNN, unrolled for 3 time steps (from <em>t </em>= 1 to <em>t </em>= 3). Label each node with the corresponding hidden unit and each edge with the corresponding weight. Note that it includes the initial hidden states for both the forward and backward RNNs.</li>

    </ol>Figure 1 – Computational graph of the bidirectional RNN unrolled for three timesteps.

    <ol start="2">

     <li>Using total derivatives we can express the gradients ∇ <sub>(<em>f</em>)</sub><em>L </em>and ∇ <sub>(<em>b</em>)</sub><em>L </em>recursively in terms of</li>

    </ol><em>h</em><em>t                                      </em><em>h</em><em>t</em>∇ (<em><sub>f</sub></em><sub>) </sub><em>L </em>and ∇ (<em><sub>b</sub></em><sub>) </sub><em>L </em>as follows:<em><sup>h</sup></em><em>t</em>+1                                  <em><sup>h</sup></em><em>t</em>−1∇<em>h</em>∇<em>h</em>(<em>f</em>)                              (<em>b</em>)<em>∂</em><em>h</em><em>t</em>+1                        <em>∂</em><em>h</em><em>t</em>−1Derive an expression for ∇<em>h</em>(<em>f</em><sub>)</sub><em>L<sub>t</sub></em>, <sup>∇</sup><em><sub>h</sub></em><em>t</em>(<em><sub>b</sub></em><sub>)</sub><em>L<sub>t</sub></em><sup>, </sup><em><sub>∂h</sub></em>(<em>tf</em><sub>) </sub><sup>and </sup><em><sub>∂h</sub></em><em>t</em>(<em><sub>b</sub></em><sub>) </sub>.<em>t</em>

    <ol start="3">

     <li>Now derive ∇<em><sub>W</sub></em>(<em><sub>f</sub></em><sub>)</sub><em>L </em>and ∇<em><sub>U</sub></em>(<em><sub>b</sub></em><sub>)</sub><em>L </em>as functions of and , respectively.</li>

    </ol><em>Hint: It might be useful to consider the contribution of the weight matrices when computing the recurrent hidden unit at a particular time </em><em>t and how those contributions might be aggregated.</em></li>

  </ol></li>

</ol>