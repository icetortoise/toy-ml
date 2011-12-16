(ns toy-ml.gmm
  (:use [incanter core distributions charts]))

(defn infinite-draw [d]
  (lazy-seq (cons (draw d)
                  (infinite-draw d))))

(def N 100)
(def out1 (take N (infinite-draw (normal-distribution 6 1))))

(def out2 (take N (infinite-draw (normal-distribution 1 1))))

(def y (map (fn [x y] (if (= 0 (draw (integer-distribution 0 2)))
                        x y))
            out1 out2))

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
    [(draw y)
     (draw y)
     (sum ($= ($= y - (mean y)) ** 2))
     (sum ($= ($= y - (mean y)) ** 2))
     0.5])
  
  (defn- compute-gamma [params x]
    (let [[mu1 mu2 s1 s2 pi] params]
      (/ ($= pi * (exp ($= (minus ($= ($= x - mu2) ** 2)) / s2)))
         (+ ($= pi * (exp ($= (minus ($= ($= x - mu2) ** 2)) / s2)))
            ($= ($= 1 - pi) * (exp ($= (minus ($= ($= x - mu1) ** 2)) / s1)))))))
  (defn- compute-gammas [params]
    (map (partial compute-gamma params) y))
  
  (defn- new-params [gamma]
    (let [mu1 ($= (sum ($= ($= 1 - gamma) * y)) / (sum ($= 1 - gamma)))
          mu2 ($= (sum ($= gamma * y)) / (sum gamma))
          s1 ($= (sum ($= ($= 1 - gamma) * ($= ($= y - mu1) ** 2)))
                 / (sum ($= 1 - gamma)))
          s2 ($= (sum ($= gamma * ($= ($= y - mu2) ** 2)))
                 / (sum gamma))
          pi ($= (sum gamma) / (count gamma))]
      [mu1 mu2 s1 s2 pi]))
  
  (defn- log-likelihoods [params]
    (defn- pdf-seq [d y]
      (map (partial pdf d) y))
    (let [[mu1 mu2 s1 s2 pi] params]
      (sum (log ($= ($= ($= 1 - pi) * (pdf-seq (normal-distribution mu1 s1) y)
                        + ($= pi * (pdf-seq (normal-distribution mu2 s2) y))))))))
  
  (defn- run-em [params]
    (let [gammas (compute-gammas params)]
      (lazy-seq (cons {:params params
                       :gammas gammas
                       :log-likelihoods (log-likelihoods params)}
                      (run-em (new-params gammas))))))
  {:E compute-gammas :M new-params
   :infinite-run (fn [] (run-em (params-init)))})