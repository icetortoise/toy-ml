(ns toy-ml.sampling
  (:use [toy-ml core])
  (:use [incanter core charts distributions])
  (:require [incanter.stats :as stats]))

(defn- true-prob [x]
  ($= ($= 0.3 * (exp (minus ($= (x - 0.1)
                                ** 2))))
      + 
      ($= 0.7 * (exp (minus (/ ($= (x - 2)
                                   ** 2) 0.3))))))

(defn- q-sampler []
  (* 4(stats/sample-uniform 1)))
(defn- q-density [x] 1)

(defn rejection
  ([prop-sampler sample-from M]
     "(Num) -> (Num->Prob) -> Num -> Num"
  (let [x (prop-sampler)
        u ($= (stats/sample-uniform 1) * M)]
    (if (< u (true-prob x))
      x
      (recur prop-sampler sample-from M))))
  ([prop-sampler sample-from M size]
     (repeatedly size #(rejection prop-sampler sample-from M))))

(defn rejection-chart []
  (histogram (rejection q-sampler true-prob 0.72 10000)
             :density true :nbins 20))

(defn importance
  ([prop-sampler prop-density sample-from]
     "(Num) ->(Num -> Prob) -> (Num->Prob) -> [Num]"
     (let [s (prop-sampler)
           w (/ (sample-from s) (prop-density s))]
       [s w]))
  ([prop-sampler prop-density sample-from size]
     (repeatedly size #(importance prop-sampler prop-density sample-from))))

(defn- ratio [n weight]
  (let [x (* n weight)]
    (if (> (- x (clojure.contrib.math/floor x))
           0.5)
      (clojure.contrib.math/ceil x)
      (clojure.contrib.math/floor x))))

(defn SIR [prop-sampler prop-density sample-from
           n-sample]
  (let [sample-weight-mat (matrix (importance prop-sampler
                                              prop-density
                                              sample-from
                                              n-sample))
        samples ($ 0 sample-weight-mat)
        weights (div ($ 1 sample-weight-mat)
                     (sum ($ 1 sample-weight-mat)))]
    (flatten (map (fn [s w]
                    (repeat (ratio n-sample w) s))
                  samples weights))))

(defn SIR-chart []
  (histogram (SIR q-sampler q-density true-prob 10000)
             :density true :nbins 20))


(defn gaussian-mix [x]
  (+ (* 0.3 (pdf (normal-distribution 3 (sqrt 10)) x))
     (* 0.7 (pdf (normal-distribution 10 (sqrt 3)) x))))
  
(defn q-dis []
  (normal-distribution 5 10))

(defn MH-independent [true-den prop-dis init]
  (let [s (draw prop-dis)
        alpha (min 1 (/ (* (true-den s) (pdf prop-dis init))
                        (* (true-den init) (pdf prop-dis s))))
        u (draw (uniform-distribution))
        value (if (< u alpha) s init)]
    (lazy-seq (cons value
                    (MH-independent true-den prop-dis value)))))


(defn q-rw-dis []
  (normal-distribution 0 10))

(defn MH-random-walk [true-den rw-dis init]
  (let [offset (draw rw-dis)
        alpha (min 1 (/ (true-den (+ init offset) )
                        (true-den init)))
        u (draw (uniform-distribution))
        value (if (< u alpha) (+ init offset) init)]
    (lazy-seq (cons value
                    (MH-independent true-den rw-dis value)))))

(defn MH-chart [MH]
  (histogram (take 100000 (MH gaussian-mix (q-dis) (draw (q-dis))))
             :density true :nbins 100))

;; gibbs implementation is delayed to the discussion of graphical models