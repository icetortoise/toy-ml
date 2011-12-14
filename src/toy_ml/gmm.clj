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

(defn params-init [y]
  [(draw y)
   (draw y)
   (sum ($= ($= y - (mean y)) ** 2))
   (sum ($= ($= y - (mean y)) ** 2))
   0.5])

(defn gmm-em-seq [y]
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
  (defn- run-em [params]
    (let [gammas (compute-gammas params)]
      (lazy-seq (cons {:params params
                       :gammas gammas}
                      (run-em (new-params gammas))))))
  (run-em (params-init y)))