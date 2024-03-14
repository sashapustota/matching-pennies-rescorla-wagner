data {
  int<lower=1> nTrials;
  int<lower=1, upper=2> choice[nTrials];
  int<lower=0, upper=1> reward[nTrials];
}

parameters {
  real<lower=0, upper=1> alpha;
  real<lower=0, upper=10> tau;
}

model {
  alpha ~ uniform(0, 1); // consistent across people & contexts
  tau ~ uniform(0, 10);
  
  vector[2] v = rep_vector(0.5, 2); // Initial values
  real pe;
  
  for (t in 1:nTrials) {
    // categorical_logit is designed to take in a vector of log odds and automatically convert it into a probability distribution over the categories.
    choice[t] ~ categorical_logit(tau * v); // we use categorical logit here to describe the likelihood of observed choices given he parameters, when we fit the model to the data. It handles the log odds directly.
    pe = reward[t] - v[choice[t]];
    v[choice[t]] += alpha * pe;
  }
}

generated quantities {
  int<lower=1, upper=2> choice_sim[nTrials]; // defines an array choice_sim with length nTrials
  int<lower=1, upper=2> choice_sim_prior[nTrials];
  vector[2] v_sim = rep_vector(0.5, 2); // making a vector of length 2, with 0.5 in both places
  vector[2] v_sim_prior = rep_vector(0.5, 2);
  real <lower=0, upper=1> alpha_prior;
  real <lower=0> tau_prior;
  
  alpha_prior = uniform_rng(0,1);
  tau_prior = uniform_rng(0,10);
  
  for (t in 1:nTrials) {
    vector[2] probs = softmax(tau * v_sim); // defines a vector of length 2, calculates the softmax of the product tau*v_sim, fills out the two elements (which together sum to 1)
    choice_sim[t] = categorical_rng(probs); // catgorical_rng randomly selects one of the two choices based on the probabilities contained in the probs vector (e.g. if probs = (0.3,0.7) there is 30% chance of choosing option 1, 70% chance of choosing option 2, reflecting the model's learned preferences up to trial t.
    // we btw use categorical_rng here instead of categorical_logit (as in the model block) bc we're interested here in generating new data (simulate choices) from the model based on the posterior distributions.
    
    real pe_sim = reward[t] - v_sim[choice[t]];
    v_sim[choice[t]] += alpha * pe_sim; // updates only the expected value corresponding to the chosen option
  }
  
    for (t in 1:nTrials) {
      vector[2] probs = softmax(tau * v_sim_prior); // defines a vector of length 2, calculates the softmax of the product tau*v_sim, fills out the two elements (which together sum to 1)
      choice_sim_prior[t] = categorical_rng(probs); // catgorical_rng randomly selects one of the two choices based on the probabilities contained in the probs vector (e.g. if probs = (0.3,0.7) there is 30% chance of choosing option 1, 70% chance of choosing option 2, reflecting the model's learned preferences up to trial t.
     
    
      real pe_sim = reward[t] - v_sim_prior[choice[t]];
      v_sim_prior[choice[t]] += alpha * pe_sim; // updates only the expected value corresponding to the chosen option
  }
}
