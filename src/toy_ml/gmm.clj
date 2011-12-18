(ns toy-ml.gmm
  (:use [incanter core distributions charts]))

(defn infinite-draw [d]
  (lazy-seq (cons (draw d)
                  (infinite-draw d))))

(defn- gen-gaussians [m-s-list]
  (vec (map (fn [[mean sigma]]
              (normal-distribution mean sigma))
            m-s-list)))

(defn generate-mixture [N mean-sigma-list]
  (let [distributions (gen-gaussians mean-sigma-list)]
        (for [n (range N)]
          (draw (nth distributions
                     (draw (integer-distribution
                            (count distributions))))))))

(defn GMM-EM [y n-mixture]
  (defn- params-init []
    [(repeatedly n-mixture (fn [] (draw y)))
     (repeat n-mixture (sum ($= ($= y - (mean y)) ** 2)))
     (repeat n-mixture (/ 1 n-mixture))])
  
  (defn- compute-gamma [params x]
    (let [[mu-coll sigma-coll pi-coll] params
          gamma-seq (map (fn [mu sigma pi]
                           ($= pi * (exp ($= (minus ($= ($= x - mu) ** 2)) / sigma))))
                         mu-coll sigma-coll pi-coll)
          gamma-sum (sum gamma-seq)]
      (map (fn [g] (/ g gamma-sum)) gamma-seq)))
  (defn- compute-gammas [params]
    (map (partial compute-gamma params) y))
  
  (defn- new-params [gammas]
    (let [mat-gammas ((comp trans matrix) gammas)
          y-trans (trans y)
          mu-coll (map (fn [gamma]
                         ($= (sum ($= gamma * y-trans)) / (sum gamma)))
                       mat-gammas)
          sigma-coll (map (fn [gamma mu]
                            ($= (sum ($= gamma * ($= ($= y-trans - mu) ** 2)))
                                / (sum gamma)))
                          mat-gammas mu-coll)
          pi-coll (map (fn [gamma]
                         ($= (sum gamma) / (count gamma)))
                       mat-gammas)]
      [mu-coll sigma-coll pi-coll]))
  
  (defn- log-likelihoods [params]
    (defn- pdf-seq [d y]
      (map (partial pdf d) y))
    (let [[mu-coll sigma-coll pi-coll] params]
      (sum (vectorize (map (fn [mu sigma pi]
                             ($= pi * (log (pdf-seq (normal-distribution mu sigma) y))))
                           mu-coll sigma-coll pi-coll)))))
  
  (defn- run-em [params]
    (let [gammas (compute-gammas params)]
      (lazy-seq (cons {:params params
                       :gammas gammas
                       :log-likelihoods (log-likelihoods params)}
                      (run-em (new-params gammas))))))
  {:E compute-gammas :M new-params
   :infinite-run (fn [] (run-em (params-init)))})