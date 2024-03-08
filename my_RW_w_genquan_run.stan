data {
  int<lower=1> nTrials;
  int<lower=1, upper=2> choice[nTrials];
  int<lower=-1, upper=1> reward[nTrials];
}

parameters {
  real<lower=0, upper=1> alpha;
  real<lower=0, upper=10> tau;
}

model {
  alpha ~ beta(1, 1);
  tau ~ uniform(0, 10);
  
  vector[2] v = rep_vector(0.5, 2); // Initial values
  real pe;
  
  for (t in 2:nTrials) {
    choice[t] ~ categorical_logit(tau * v);
    pe = reward[t] - v[choice[t]];
    v[choice[t]] += alpha * pe;
  }
}

generated quantities {
  int<lower=1, upper=2> choice_sim[nTrials];
  vector[2] v_sim = rep_vector(0.5, 2);
  
  choice_sim[1] = 1; // temporarily hard-coding a valid value for the first trial bc otherwise it wont run :(
  
  for (t in 2:nTrials) {
    vector[2] probs = softmax(tau * v_sim);
    choice_sim[t] = categorical_rng(probs); // 'probs' is valid
    
    real pe_sim = reward[t] - v_sim[choice_sim[t]];
    v_sim[choice_sim[t]] += alpha * pe_sim;
  }
}
