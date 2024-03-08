data {
  int<lower=1> nTrials;
  int<lower=1> nSubjects;
  int<lower=1, upper=2> choice[nTrials];
  int<lower=-1, upper=1> reward[nTrials];
  // This is the data we simulate outside of Stan - here we just provide pointers
  // to what data should be used and what it should look like.
}

parameters {
  real<lower=0, upper=1> alpha_mu;
  real<lower=0, upper=3> tau_mu;
  real<lower=0> alpha_sd;
  real<lower=0> tau_sd;
  
  real<lower=0, upper=1> alpha[nSubjects];
  real<lower=0, upper=3> tau[nSubjects];
}

model {
  
  alpha_sd ~ cauchy(0,1);
  tau_sd ~ cauchy(0,3);
  alpha ~ normal(alpha_mu, alpha_sd);
  tau ~ normal(tau_mu, tau_sd);
  
  for (s in 1:nSubjects) {
  
  vector[2] v;
  real pe;
  v = rep_vector(0, 2);
  
    for (t in 2:nTrials) {
      
        // int choice_index = choice[t] + 1;  // New variable to adjust indexing  
        
        // or simply choice[t] ~ categorical(softmax(tau * v))
        choice[s,t] ~ categorical_logit(tau[s] * v);
  
        // prediction error
        pe = reward[s,t] - v[choice[s,t]];
        // value update
        v[choice[s,t]] = v[choice[s,t]] + alpha[s] * pe;
    }
  }
}
