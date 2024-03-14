data {
  int<lower=1> nTrials;
  int<lower=1, upper=2> choice[nTrials];
  int<lower=0, upper=1> reward[nTrials];
  // This is the data we simulate outside of Stan - here we just provide pointers
  // to what data should be used and what it should look like.
}

parameters {
  real<lower=0, upper=1> alpha;
  real<lower=0, upper=10> tau;
}

model {
  vector[2] v;
  real pe;
  
  v = rep_vector(0.5, 2);
  
  for (t in 2:nTrials) {
    
      // int choice_index = choice[t] + 1;  // New variable to adjust indexing  
      
      // or simply choice[t] ~ categorical(softmax(tau * v))
      choice[t] ~ categorical_logit(tau * v);

      // prediction error
      pe = reward[t] - v[choice[t]];
      // value update
      v[choice[t]] = v[choice[t]] + alpha * pe;
  }
}
