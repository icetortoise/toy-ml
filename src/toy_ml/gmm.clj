(ns toy-ml.gmm
  (:use [incanter core distributions charts]
        [toy-ml core])
  (:require [incanter.stats :as stats]))

(defn infinite-draw [d]
  (lazy-seq (cons (draw d)
                  (infinite-draw d))))

(defn- mix [means sigmas n]
  (->> (map #(stats/sample-mvn n :mean %1 :sigma %2)
           means sigmas)
       (apply conj-rows)
       randomize))

(defn generate-mixture [N & {:keys [means sigmas]}]
  (let [ds (mix means sigmas (inc (/ N (count means))))]
    (to-matrix ($ (range N) :all ds))))

(defn- multivariant-normal-pdf [mean sigma x]
  (let [diff ($= x - mean)]
    ($= (exp ($= ($= (trans diff) <*> (solve sigma) <*> diff) / (minus 2)))
        / (mult (pow ($= Math/PI * 2) ($= (count x) / 2))
                (pow (det sigma) (/ 1 2))))))

(defn GMM-EM [y n-mixture]
  (defn- params-init []
    [(repeatedly n-mixture (fn [] (trans (draw y))))
     (repeat n-mixture (stats/covariance y))
     (repeat n-mixture (/ 1 n-mixture))])
  
  (defn- compute-gamma [params x]
    (let [[mu-coll sigma-coll pi-coll] params
          gamma-seq (map (fn [mu sigma pi]
                           ($= pi * (multivariant-normal-pdf mu sigma (trans x))))
                         mu-coll sigma-coll pi-coll)
          gamma-sum (sum gamma-seq)]
      (map (fn [g] (/ g gamma-sum)) gamma-seq)))
  (defn- compute-gammas [params]
    (map (partial compute-gamma params) y))
  
  (defn- new-params [gammas]
    (let [mat-gammas ((comp trans matrix) gammas)
          y-trans (trans y)
          mu-coll (map (fn [gamma]
                         (matrix ($= ($= gamma <*> y) / (sum gamma))))
                       mat-gammas)
          sigma-coll (map (fn [gamma mu]
                            (let [delta (minus y (matrix (repeat (count gamma) mu)))]
                              ($= (apply plus (map #($= %1 * %2)
                                                   gamma
                                                   (map (fn [x] ($= (trans x) <*> x))
                                                        delta)))
                                / (sum gamma))))
                          mat-gammas mu-coll)
          pi-coll (map (fn [gamma]
                         ($= (sum gamma) / (count gamma)))
                       mat-gammas)]
      [mu-coll sigma-coll pi-coll]))
  
  (defn- log-likelihoods [params]
    (let [[mu-coll sigma-coll pi-coll] params]
      (defn- ll-one [x]
        (log (sum (map (fn [mu sigma pi]
                         (multivariant-normal-pdf
                          mu sigma (trans x)))
                       mu-coll sigma-coll pi-coll))))
      (sum (map ll-one y))))
  
  (defn- run-em [params]
    (let [gammas (compute-gammas params)]
      (lazy-seq (cons {:params params
                       :gammas gammas
                       :log-likelihoods (log-likelihoods params)}
                       (run-em (new-params gammas))))))
  {:E compute-gammas :M new-params
   :infinite-run (fn [] (run-em (params-init)))})