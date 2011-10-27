(ns toy-ml.linear)
(use '(toy-ml core))
(use '(incanter core))

(defn normalize [input]
  (bind-rows (matrix -1 1 (ncol input))
             input))

(defn rand-matrix
  ([n nrow ncol]
     (rand-matrix n (matrix 0 nrow ncol)))
  ([n mat]
     (matrix (matrix-map (fn [x] (rand 1)) mat))))

(defn end-after-iter [iter]
  (fn [n] (if (>= n iter) true false)))

(defn activate [val]
  (if (> val 0) 1 0)
  )

(defn pcn
  "Takes input matrix(each colume is a data point) and target matrix(each column is a taget vector),
learning rate and a function indicating when to stop learning, return the learned weights."
  [input-mat target l-rate end-fn]
  (let [normalized (normalize input-mat)
        weights (rand-matrix 1 (nrow normalized)
                              (ncol target))]
    (loop [w weights iter 1]
      (let [ys (trans (m-map activate
                             (mmult (trans w) normalized)))]
        (if (end-fn iter) w
            (recur (plus w
                         (mult l-rate
                               (mmult normalized
                                      (minus target ys))))
                   (+ iter 1)))))))



(defn pcn-predict [input weights]
  (activate (mmult (trans weights) (normalize input))))

(defn pcn-or [input]
  (pcn-predict input (pcn (trans (matrix [[0 0] [0 1] [1 0] [1 1]]))
                          (matrix [0 1 1 1]) 0.1 (end-after-iter 200))))

(defn pcn-parity [input]
  "Will give poor result as it is not linearly seperapable"
  (if (not (= (nrow input) 3))
    (assert input "expecting a colum vector in R^3")
    (pcn-predict input (pcn (trans (matrix [[0 0 0] [0 0 1] [0 1 0] [1 0 0]
                                            [1 1 0] [1 0 1] [0 1 1] [1 1 1]]))
                            (matrix [1 0 0 0 1 1 1 0]) 0.1 (end-after-iter 200)))))