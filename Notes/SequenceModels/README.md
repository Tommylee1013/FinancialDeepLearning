### Preface

이제까지는 sequence를 Auto Correlated Process, Moving average, GARCH 등 이들과 유사한 모델을 이용해 모델링하는 것을 배워왔다.
한편, Hidden markov model, Baum–Welch Algorithm, Viterbi Algorithm, Kalman과 입자 필터링을 산출한 다른 학파가 존재한다.

이 학파는 특정 잠재 공간이 존재를 가정한다. 이는 시간에 따라 진화한다. 이 관측 불가능한 잠재 공간은 또 다른 관측 가능한 공간을 주도하고, 모든 시점이나 일정 구간의 시점에서 $Y_t$를 관측한다.
잠재 공간 $X_t$의 진화는 관측 가능한 프로세스 $Y_t$의 $X_t$에 대한 의존성과 함께 랜덤 요인에 의해 결정된다. 따라서 우리는 확률적 모델 (probabilistic model, stochastic model)을 다루고 있는 것이다. 
또한 이와 같은 모델을 State-Space Model이라고 한다. State-space model은 시간에 따른 잠재 상태의 진화에 대한 묘사와 관측 가능한 변수의 잠재 상태에 대한 의존성으로 구성된다.

